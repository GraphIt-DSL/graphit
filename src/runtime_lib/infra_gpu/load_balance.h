#ifndef GRAPHIT_GPU_LOAD_BALANCE_H
#define GRAPHIT_GPU_LOAD_BALANCE_H

#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"

namespace gpu_runtime {

template <void body(int32_t vid)>
static void __device__ vertex_set_apply(int32_t num_vertices) {
	for(int32_t vid = threadIdx.x + blockDim.x * blockIdx.x; vid < num_vertices; vid+= blockDim.x * gridDim.x) {
		body(vid);
	}
}
template <void body(int32_t vid)>
static void __global__ vertex_set_apply_kernel(int32_t num_vertices) {
	vertex_set_apply<body>(num_vertices);
} 

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier)>
void __device__ vertex_based_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	int32_t vid = threadIdx.x + blockDim.x * blockIdx.x;
	if (vid >= input_frontier.d_num_elems_input[0])
		return;
	int32_t src = input_frontier.d_sparse_queue_input[vid];
	for (int32_t eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; eid++) {
		int32_t dst = graph.d_edge_dst[eid];
		load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);
	}
}

void __host__ vertex_based_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = builtin_getVertexSetSize(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}
#define STAGE_1_SIZE (8)
#define WARP_SIZE (32)
template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier)>
static void __device__ TWCE_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	int32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	
	int32_t lane_id = thread_id % 32;
	
	__shared__ int32_t stage2_queue[CTA_SIZE];
	__shared__ int32_t stage3_queue[CTA_SIZE];
	__shared__ int32_t stage_queue_sizes[3];
	
	if (threadIdx.x == 0) {
		stage_queue_sizes[0] = 0;
		stage_queue_sizes[1] = 0;
		stage_queue_sizes[2] = 0;
	}
	__syncthreads();
	__shared__ int32_t stage2_offset[CTA_SIZE];
	__shared__ int32_t stage3_offset[CTA_SIZE];
	__shared__ int32_t stage2_size[CTA_SIZE];
	__shared__ int32_t stage3_size[CTA_SIZE];	

	int32_t total_vertices = input_frontier.d_num_elems_input[0];
	int32_t local_vertex_idx = thread_id / (STAGE_1_SIZE);
	int32_t degree;
	int32_t s1_offset;
	int32_t local_vertex;
	int32_t src_offset;
	if (local_vertex_idx < total_vertices) {
		local_vertex = input_frontier.d_sparse_queue_input[local_vertex_idx];
		// Step 1 seggrefate vertices into shared buffers
		if (threadIdx.x % (STAGE_1_SIZE) == 0) {
			degree = graph.d_get_degree(local_vertex);
			src_offset = graph.d_src_offsets[local_vertex];
			int32_t s3_size = degree/CTA_SIZE;
			degree = degree - s3_size * CTA_SIZE;
			if (s3_size > 0) {
				int32_t pos = atomicAggInc(&stage_queue_sizes[2]);
				stage3_queue[pos] = local_vertex;
				stage3_size[pos] = s3_size * CTA_SIZE;
				stage3_offset[pos] = src_offset;
			}

			int32_t s2_size = degree/WARP_SIZE;
			degree = degree - WARP_SIZE * s2_size;
			if (s2_size > 0) {
				int32_t pos = atomicAggInc(&stage_queue_sizes[1]);
				stage2_queue[pos] = local_vertex;
				stage2_offset[pos] = s3_size * CTA_SIZE + src_offset;
				stage2_size[pos] = s2_size * WARP_SIZE;
			}
			s1_offset = s3_size * CTA_SIZE + s2_size * WARP_SIZE + src_offset;
		}
	} else 
		local_vertex = -1;
	__syncthreads();
	degree = __shfl_sync((uint32_t)-1, degree, (lane_id / STAGE_1_SIZE) * STAGE_1_SIZE, 32);
	s1_offset = __shfl_sync((uint32_t)-1, s1_offset, (lane_id / STAGE_1_SIZE) * STAGE_1_SIZE, 32);
	local_vertex = __shfl_sync((uint32_t)-1, local_vertex, (lane_id / STAGE_1_SIZE) * STAGE_1_SIZE, 32);

	if (local_vertex_idx < total_vertices) {
		// STAGE 1
		for (int32_t neigh_id = s1_offset + (lane_id % STAGE_1_SIZE); neigh_id < degree + s1_offset; neigh_id += STAGE_1_SIZE) {
			int32_t dst = graph.d_edge_dst[neigh_id];
			load_balance_payload(graph, local_vertex, dst, neigh_id, input_frontier, output_frontier);	
		}

	}
	__syncwarp();
	// STAGE 2 -- stage 2 is dynamically balanced
	while(1) {
		int32_t to_process;
		if (lane_id == 0) {
			to_process = atomicSub(&stage_queue_sizes[1], 1) - 1;
		}
		to_process = __shfl_sync((uint32_t)-1, to_process, 0, 32);
		if (to_process < 0)
			break;
		local_vertex = stage2_queue[to_process];
		degree = stage2_size[to_process];
		int32_t s2_offset = stage2_offset[to_process];
		for (int32_t neigh_id = s2_offset + (lane_id); neigh_id < degree + s2_offset; neigh_id += WARP_SIZE) {
			int32_t dst = graph.d_edge_dst[neigh_id];
			load_balance_payload(graph, local_vertex, dst, neigh_id, input_frontier, output_frontier);	
		}
		
	}	
	// STAGE 3 -- all threads have to do all, no need for LB
	for (int32_t wid = 0; wid < stage_queue_sizes[2]; wid++) {
		local_vertex = stage3_queue[wid];
		degree = stage3_size[wid];
		int32_t s3_offset = stage3_offset[wid];
		for (int32_t neigh_id = s3_offset + (threadIdx.x); neigh_id < degree + s3_offset; neigh_id += CTA_SIZE) {
			int32_t dst = graph.d_edge_dst[neigh_id];
			load_balance_payload(graph, local_vertex, dst, neigh_id, input_frontier, output_frontier);	
		}	
	}
}
void __host__ TWCE_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = builtin_getVertexSetSize(frontier) * STAGE_1_SIZE;
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

}

#endif
