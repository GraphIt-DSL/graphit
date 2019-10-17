#ifndef GRAPHIT_GPU_LOAD_BALANCE_H
#define GRAPHIT_GPU_LOAD_BALANCE_H

#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

namespace gpu_runtime {

template <typename EdgeWeightType>
using load_balance_payload_type = void (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier);


// VERTEX SET APPLY FUNCTIONS
template <typename AccessorType, void body(int32_t vid)>
static void __device__ vertex_set_apply(VertexFrontier &frontier) {
	int32_t total_vertices = AccessorType::getSize(frontier);
	for(int32_t vidx = threadIdx.x + blockDim.x * blockIdx.x; vidx < total_vertices; vidx += blockDim.x * gridDim.x) {
		int32_t vid = AccessorType::getElement(frontier, vidx);
		body(vid);
	}
}
template <typename AccessorType, void body(int32_t vid)>
static void __global__ vertex_set_apply_kernel(VertexFrontier frontier) {
	vertex_set_apply<AccessorType, body>(frontier);
} 

// VERTEX BASED LOAD BALANCE FUNCTIONS
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __device__ vertex_based_load_balance(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier, unsigned int cta_id, unsigned int num_cta) {
	int32_t vid = threadIdx.x + blockDim.x * cta_id;
	if (vid >= AccessorType::getSize(input_frontier))
		return;
	int32_t src = AccessorType::getElement(input_frontier, vid);
	for (int32_t eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; eid++) {
		if (src_filter(src) == false)
			break;
		int32_t dst = graph.d_edge_dst[eid];
		load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);
	}
}
template <typename AccessorType>
void __host__ vertex_based_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}
template <typename AccessorType>
void __device__ vertex_based_load_balance_info_device(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeDevice(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ vertex_based_load_balance_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	vertex_based_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ vertex_based_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	vertex_based_load_balance_info<AccessorType>(input_frontier, num_cta, cta_size);
	vertex_based_load_balance_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __device__ vertex_based_load_balance_device(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	vertex_based_load_balance_info_device<AccessorType>(input_frontier, num_cta, cta_size);
	this_grid().sync();
	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		vertex_based_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);	
		__syncthreads();
	}
	this_grid().sync();
}

// EDGE_ONLY LOAD BALANCE FUNCTIONS

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
static void __device__ edge_only_load_balance(GraphT<EdgeWeightType> &graph, VertexFrontier input_frontier, VertexFrontier output_frontier, unsigned int cta_id, unsigned int total_cta) {
	int32_t thread_id = blockDim.x * cta_id + threadIdx.x;
	int32_t total_threads = blockDim.x * total_cta;
	for (int32_t eid = thread_id; eid < graph.num_edges; eid += total_threads) {
		int32_t src = graph.d_edge_src[eid];
		if (src_filter(src) == true) {
			int32_t dst = graph.d_edge_dst[eid];
			load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);	
		}
	}		
}
template <typename AccessorType>
void __host__ edge_only_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	num_cta = NUM_CTA;
	cta_size = CTA_SIZE;
}
template <typename AccessorType>
void __device__ edge_only_load_balance_info_device(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	num_cta = NUM_CTA;
	cta_size = CTA_SIZE;
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ edge_only_load_balance_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	edge_only_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ edge_only_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	edge_only_load_balance_info<AccessorType>(input_frontier, num_cta, cta_size);
	edge_only_load_balance_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __device__ edge_only_load_balance_device(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	vertex_based_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);	
	this_grid().sync();
}

// TWCE LOAD BALANCE FUNCTIONS
#define STAGE_1_SIZE (8)
#define WARP_SIZE (32)
template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
static void __device__ TWCE_load_balance(GraphT<EdgeWeightType> &graph, VertexFrontier input_frontier, VertexFrontier output_frontier, unsigned int cta_id, unsigned int total_cta) {
	int32_t thread_id = blockDim.x * cta_id + threadIdx.x;
	
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

	int32_t total_vertices = AccessorType::getSize(input_frontier);
	int32_t local_vertex_idx = thread_id / (STAGE_1_SIZE);
	int32_t degree;
	int32_t s1_offset;
	int32_t local_vertex;
	int32_t src_offset;
	if (local_vertex_idx < total_vertices) {
		local_vertex = AccessorType::getElement(input_frontier, local_vertex_idx);
		// Step 1 seggregate vertices into shared buffers
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
			if (src_filter(local_vertex) == false)
				break;
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
			if (src_filter(local_vertex) == false)
				break;
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
			if (src_filter(local_vertex) == false)
				break;
			int32_t dst = graph.d_edge_dst[neigh_id];
			load_balance_payload(graph, local_vertex, dst, neigh_id, input_frontier, output_frontier);	
		}	
	}
}
template <typename AccessorType>
void __host__ TWCE_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier) * STAGE_1_SIZE;
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}
template <typename AccessorType>
void __device__ TWCE_load_balance_info_device(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSize(frontier) * STAGE_1_SIZE;
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ TWCE_load_balance_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	TWCE_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ TWCE_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	TWCE_load_balance_info<AccessorType>(input_frontier, num_cta, cta_size);
	TWCE_load_balance_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __device__ TWCE_load_balance_device(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	TWCE_load_balance_info_device<AccessorType>(input_frontier, num_cta, cta_size);
	this_grid().sync();
	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		TWCE_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);	
		__syncthreads();
	}
	this_grid().sync();
}
}

#endif
