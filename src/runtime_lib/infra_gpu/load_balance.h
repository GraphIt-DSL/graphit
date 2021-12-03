#ifndef GRAPHIT_GPU_LOAD_BALANCE_H
#define GRAPHIT_GPU_LOAD_BALANCE_H

#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"
#include "infra_gpu/gpu_priority_queue.h"
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

/*
	int32_t total_vertices = AccessorType::getSize(input_frontier);
	for (int32_t vidx = threadIdx.x + blockDim.x * cta_id; vidx < total_vertices; vidx += num_cta * blockDim.x) {
		int32_t src = AccessorType::getElement(input_frontier, vidx);
		for (int32_t eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; eid++) {
			if (src_filter(src) == false)
				break;
			int32_t dst = graph.d_edge_dst[eid];
			load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);
		}	
	}
*/
}
template <typename AccessorType>
void __host__ vertex_based_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {

	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;

	//num_cta = NUM_CTA;
	cta_size = CTA_SIZE;
}
template <typename AccessorType>
void __device__ vertex_based_load_balance_info_device(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSize(frontier);
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

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
static void __device__ edge_only_load_balance_blocked(GraphT<EdgeWeightType> &graph, VertexFrontier input_frontier, VertexFrontier output_frontier, unsigned int cta_id, unsigned int total_cta, int32_t index) {
	int32_t thread_id = blockDim.x * cta_id + threadIdx.x;
	int32_t total_threads = blockDim.x * total_cta;
	int32_t starting_edge = index == 0?0:graph.d_bucket_sizes[index-1];
	int32_t ending_edge = graph.d_bucket_sizes[index];
	for (int32_t eid = thread_id + starting_edge; eid < ending_edge; eid += total_threads) {
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
void __global__ edge_only_load_balance_blocked_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	for (int32_t index = 0; index < graph.num_buckets; index++) {
		edge_only_load_balance_blocked<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x, index);
		__syncthreads();
	}
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ edge_only_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	edge_only_load_balance_info<AccessorType>(input_frontier, num_cta, cta_size);
	edge_only_load_balance_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __host__ edge_only_load_balance_blocked_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta = NUM_CTA;
	int32_t cta_size = CTA_SIZE;
	edge_only_load_balance_blocked_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);
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

static void __device__ TWCE_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier, unsigned int cta_id, unsigned int total_cta) {
	int32_t thread_id = blockDim.x * cta_id + threadIdx.x;
	
	int32_t lane_id = thread_id % 32;
	
	__shared__ int32_t stage2_queue[CTA_SIZE];
	__shared__ int32_t stage3_queue[CTA_SIZE];
	__shared__ int32_t stage_queue_sizes[3];
	
	if (threadIdx.x < 3) {
		stage_queue_sizes[threadIdx.x] = 0;
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
		degree = graph.d_get_degree(local_vertex);
		src_offset = graph.d_src_offsets[local_vertex];
		int32_t s3_size = degree/CTA_SIZE;
		degree = degree - s3_size * CTA_SIZE;
		if (s3_size > 0) {
			if (threadIdx.x % (STAGE_1_SIZE) == 0) {
				int32_t pos = atomicAggInc(&stage_queue_sizes[2]);
				stage3_queue[pos] = local_vertex;
				stage3_size[pos] = s3_size * CTA_SIZE;
				stage3_offset[pos] = src_offset;
			}
		}

		int32_t s2_size = degree/WARP_SIZE;
		degree = degree - WARP_SIZE * s2_size;
		if (s2_size > 0) {
			if (threadIdx.x % (STAGE_1_SIZE) == 0) {
				int32_t pos = atomicAggInc(&stage_queue_sizes[1]);
				stage2_queue[pos] = local_vertex;
				stage2_offset[pos] = s3_size * CTA_SIZE + src_offset;
				stage2_size[pos] = s2_size * WARP_SIZE;
			}
		}
		s1_offset = s3_size * CTA_SIZE + s2_size * WARP_SIZE + src_offset;
	} else 
		local_vertex = -1;
	__syncthreads();
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

// CM load balance functions
int32_t __device__ binary_search_upperbound(int32_t *array, int32_t len, int32_t key){
	int32_t s = 0;
	while(len>0){
		int32_t half = len>>1;
		int32_t mid = s + half;
		if(array[mid] > key){
			len = half;
		}else{
			s = mid+1;
			len = len-half-1;
		}
	}
	return s;
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __device__ CM_load_balance(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier, unsigned int cta_id, unsigned int num_cta) {

	__shared__ int32_t sm_idx[CTA_SIZE], sm_deg[CTA_SIZE], sm_loc[CTA_SIZE];
	int32_t thread_id = threadIdx.x + blockDim.x * cta_id;
	int32_t tot_size = AccessorType::getSize(input_frontier);

        int32_t deg, index, src_idx;
        if(thread_id < tot_size) {
		index = AccessorType::getElement(input_frontier, thread_id);
                deg = graph.d_get_degree(index);

		sm_idx[threadIdx.x] = index;
                sm_deg[threadIdx.x] = deg;
                sm_loc[threadIdx.x] = graph.d_src_offsets[index];
        } else {
                deg = 0;
                sm_deg[threadIdx.x] = deg;
        }

        int32_t lane = (threadIdx.x & 31);
        int32_t offset = 0;
	
	// prefix sum
	int32_t cosize = blockDim.x;
	int32_t tot_deg;
	int32_t phase = threadIdx.x;
	int32_t off=32;

	for(int32_t d=2; d<=32; d<<=1) {
		int32_t temp = __shfl_up_sync((uint32_t)-1, deg, d/2);
		if (lane % d == d - 1) deg += temp;
	}
	sm_deg[threadIdx.x] = deg;

	for(int32_t d=cosize>>(1+5); d>0; d>>=1){
		__syncthreads();
		if(phase<d){
			int32_t ai = off*(2*phase+1)-1;
			int32_t bi = off*(2*phase+2)-1;
			sm_deg[bi] += sm_deg[ai];
		}
		off<<=1;
	}

	__syncthreads();
	tot_deg = sm_deg[cosize-1];
	__syncthreads();
	if(!phase) sm_deg[cosize-1]=0;
	__syncthreads();

	for(int32_t d=1; d<(cosize>>5); d<<=1){
		off >>=1;
		__syncthreads();
		if(phase<d){
			int32_t ai = off*(2*phase+1)-1;
			int32_t bi = off*(2*phase+2)-1;

			int32_t t = sm_deg[ai];
			sm_deg[ai]  = sm_deg[bi];
			sm_deg[bi] += t;
		}
	}
	__syncthreads();
	deg = sm_deg[threadIdx.x];
	__syncthreads();
	for(int32_t d=32; d>1; d>>=1) {
		int32_t temp_big = __shfl_down_sync((uint32_t)-1, deg, d/2);
		int32_t temp_small = __shfl_up_sync((uint32_t)-1, deg, d/2);
		if (lane % d == d/2 - 1) deg = temp_big;
		else if(lane % d == d - 1) deg += temp_small;
	}
	sm_deg[threadIdx.x] = deg;
	__syncthreads();
	
	// compute
        int32_t width = thread_id - threadIdx.x + blockDim.x;
        if(tot_size < width) width = tot_size;
        width -= thread_id - threadIdx.x;

        for(int32_t i=threadIdx.x; i<tot_deg; i+=blockDim.x) {
                int32_t id = binary_search_upperbound(&sm_deg[offset], width, i)-1;

                if(id >= width) continue;
                src_idx = sm_idx[offset + id];
		if (src_filter(src_idx) == false)
			continue;
                int32_t ei = sm_loc[offset + id] + i - sm_deg[offset + id];
                int32_t dst_idx = graph.d_edge_dst[ei];
		load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
        }
}
template <typename AccessorType>
void __host__ CM_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}
template <typename AccessorType>
void __device__ CM_load_balance_info_device(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSize(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ CM_load_balance_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	CM_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ CM_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	CM_load_balance_info<AccessorType>(input_frontier, num_cta, cta_size);
	CM_load_balance_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __device__ CM_load_balance_device(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	CM_load_balance_info_device<AccessorType>(input_frontier, num_cta, cta_size);
	this_grid().sync();
	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		CM_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);	
		__syncthreads();
	}
	this_grid().sync();
}


// WM load balance functions
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __device__ WM_load_balance(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier, unsigned int cta_id, unsigned int num_cta) {

	__shared__ int32_t sm_idx[CTA_SIZE], sm_deg[CTA_SIZE], sm_loc[CTA_SIZE];
	int32_t thread_id = threadIdx.x + blockDim.x * cta_id;
	int32_t tot_size = AccessorType::getSize(input_frontier);

        int32_t deg, index, src_idx;
        if(thread_id < tot_size) {
		index = AccessorType::getElement(input_frontier, thread_id);
                deg = graph.d_get_degree(index);

		sm_idx[threadIdx.x] = index;
                sm_deg[threadIdx.x] = deg;
                sm_loc[threadIdx.x] = graph.d_src_offsets[index];
        } else {
                deg = 0;
                sm_deg[threadIdx.x] = deg;
        }

        // prefix sum
        int32_t lane = (threadIdx.x&31);
        int32_t offset = threadIdx.x - lane;
        for(int32_t d=1; d<32; d<<=1) {
                int32_t temp = __shfl_up_sync((uint32_t)-1, deg, d);
                if (lane >= d) deg += temp;
        }
        int32_t tot_deg = __shfl_sync((uint32_t)-1, deg, 31);
        if(lane == 31) deg = 0;
        sm_deg[offset + ((lane+1)&31)] = deg;
        __syncthreads();

        // compute
        int32_t width = thread_id - lane + 32;
        if(tot_size < width) width = tot_size;
        width -= thread_id - lane;

        for(int32_t i=lane; i<tot_deg; i+=32) {
                int32_t id = binary_search_upperbound(&sm_deg[offset], width, i)-1;

                src_idx = sm_idx[offset + id];
		if (src_filter(src_idx) == false)
			continue;

                int32_t ei = sm_loc[offset + id] + i - sm_deg[offset + id];
                int32_t dst_idx = graph.d_edge_dst[ei];
		load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
        }
}
template <typename AccessorType>
void __host__ WM_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}
template <typename AccessorType>
void __device__ WM_load_balance_info_device(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSize(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ WM_load_balance_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	WM_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ WM_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	WM_load_balance_info<AccessorType>(input_frontier, num_cta, cta_size);
	WM_load_balance_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __device__ WM_load_balance_device(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t num_cta, cta_size;
	WM_load_balance_info_device<AccessorType>(input_frontier, num_cta, cta_size);
	this_grid().sync();
	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		WM_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);	
		__syncthreads();
	}
	this_grid().sync();
}

//TWCE load balance functions
#define MID_BIN (32)
#define LARGE_BIN (CTA_SIZE)

template <typename EdgeWeightType, typename AccessorType>
void __device__ TWC_split_frontier (GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, unsigned int cta_id, unsigned int num_cta) {
        int32_t thread_id = threadIdx.x + blockDim.x * cta_id;
        int32_t tot_size = AccessorType::getSize(input_frontier);
	int32_t idx, deg;
	if(thread_id < tot_size) {
		idx = AccessorType::getElement(input_frontier, thread_id);
		deg = graph.d_get_degree(idx);
		if(deg < MID_BIN) {
			int32_t k = atomicAggInc(&graph.twc_bin_sizes[0]);
			graph.twc_small_bin[k] = idx;
		} else if(deg < LARGE_BIN) {
			int32_t k = atomicAggInc(&graph.twc_bin_sizes[1]);
			graph.twc_mid_bin[k] = idx;
		} else {
			int32_t k = atomicAggInc(&graph.twc_bin_sizes[2]);
			graph.twc_large_bin[k] = idx;
		}
	}	
}
template <typename EdgeWeightType, typename AccessorType>
void __global__ TWC_split_frontier_kernel (GraphT<EdgeWeightType> graph, VertexFrontier input_frontier) {
	TWC_split_frontier<EdgeWeightType, AccessorType> (graph, input_frontier, blockIdx.x, gridDim.x);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __device__ TWC_small_bin (GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier, unsigned int cta_id, unsigned int num_cta) {

	__shared__ int32_t sm_idx[CTA_SIZE], sm_deg[CTA_SIZE], sm_loc[CTA_SIZE];
	int32_t thread_id = threadIdx.x + blockDim.x * cta_id;
	int32_t tot_size = graph.twc_bin_sizes[0];

        int32_t deg, index, src_idx;
        if(thread_id < tot_size) {
		index = graph.twc_small_bin[thread_id];
                deg = graph.d_get_degree(index);

		sm_idx[threadIdx.x] = index;
                sm_deg[threadIdx.x] = deg;
                sm_loc[threadIdx.x] = graph.d_src_offsets[index];
        } else {
                deg = 0;
                sm_deg[threadIdx.x] = deg;
        }

        // prefix sum
        int32_t lane = (threadIdx.x&31);
        int32_t offset = threadIdx.x - lane;
        for(int32_t d=1; d<32; d<<=1) {
                int32_t temp = __shfl_up_sync((uint32_t)-1, deg, d);
                if (lane >= d) deg += temp;
        }
        int32_t tot_deg = __shfl_sync((uint32_t)-1, deg, 31);
        if(lane == 31) deg = 0;
        sm_deg[offset + ((lane+1)&31)] = deg;
        __syncthreads();

        // compute
        int32_t width = thread_id - lane + 32;
        if(tot_size < width) width = tot_size;
        width -= thread_id - lane;

        for(int32_t i=lane; i<tot_deg; i+=32) {
                int32_t id = binary_search_upperbound(&sm_deg[offset], width, i)-1;

                src_idx = sm_idx[offset + id];
		if (src_filter(src_idx) == false)
			continue;

                int32_t ei = sm_loc[offset + id] + i - sm_deg[offset + id];
                int32_t dst_idx = graph.d_edge_dst[ei];
		load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
        }
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ TWC_small_bin_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	TWC_small_bin<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
	
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __device__ TWC_mid_bin (GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier, unsigned int cta_id, unsigned int num_cta) {
	int32_t vid = (threadIdx.x + blockDim.x * cta_id)/MID_BIN;
	int32_t tot_size = graph.twc_bin_sizes[1];
	
	if (vid >= tot_size)
		return;

	int32_t src = graph.twc_mid_bin[vid];
	for (int32_t eid = graph.d_src_offsets[src]+(threadIdx.x%MID_BIN); eid < graph.d_src_offsets[src+1]; eid+=MID_BIN) {
		if (src_filter(src) == false)
			break;
		int32_t dst = graph.d_edge_dst[eid];
		load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);
	}
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ TWC_mid_bin_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	TWC_mid_bin<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
	
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __device__ TWC_large_bin (GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier, unsigned int cta_id, unsigned int num_cta) {
	int32_t vid = (threadIdx.x + blockDim.x * cta_id)/LARGE_BIN;
	int32_t tot_size = graph.twc_bin_sizes[2];
	if (vid >= tot_size)
		return;
	int32_t src = graph.twc_large_bin[vid];
	for (int32_t eid = graph.d_src_offsets[src]+(threadIdx.x%LARGE_BIN); eid < graph.d_src_offsets[src+1]; eid+=LARGE_BIN) {
		if (src_filter(src) == false)
			break;
		int32_t dst = graph.d_edge_dst[eid];
		load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);
	}
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ TWC_large_bin_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	TWC_large_bin<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
	
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ TWC_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	cudaMemset(graph.twc_bin_sizes, 0, sizeof(int32_t) * 3);
	int num_threads = AccessorType::getSizeHost(input_frontier);	
	int num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	int cta_size = CTA_SIZE;
	TWC_split_frontier_kernel<EdgeWeightType, AccessorType><<<num_cta, cta_size>>>(graph, input_frontier);
	int32_t twc_bin_sizes[3];
	cudaMemcpy(twc_bin_sizes, graph.twc_bin_sizes, 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);
	num_threads = twc_bin_sizes[0];	
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	if (num_cta)
		TWC_small_bin_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier); 
	num_threads = twc_bin_sizes[1] * MID_BIN;	
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	if (num_cta)
		TWC_mid_bin_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier); 
	num_threads = twc_bin_sizes[2] * LARGE_BIN;	
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	if (num_cta)
		TWC_large_bin_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier); 	
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __device__ TWC_load_balance_device(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int32_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id < 3) {
		graph.twc_bin_sizes[thread_id] = 0;
	}	
	this_grid().sync();

	int num_threads = AccessorType::getSize(input_frontier);	
	int num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;

	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		TWC_split_frontier<EdgeWeightType, AccessorType>(graph, input_frontier, cta_id, num_cta);
		__syncthreads();
	}

	this_grid().sync();	

	num_threads = graph.twc_bin_sizes[0];	
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	
	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		TWC_small_bin<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);
		__syncthreads();
	}

	num_threads = graph.twc_bin_sizes[1] * MID_BIN;
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;

	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		TWC_mid_bin<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);
		__syncthreads();
	}

	num_threads = graph.twc_bin_sizes[2] * LARGE_BIN;
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;

	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {
		TWC_large_bin<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);
		__syncthreads();
	}
	
	this_grid().sync();
}

// STRICT LOAD BALANCE FUNCTIONS

#define NNZ_PER_BLOCK (CTA_SIZE)
#define STRICT_SM_SIZE (CTA_SIZE)
#define PREFIX_BLK (CTA_SIZE)

template <typename AccessorType, typename EdgeWeightType>
void __device__ strict_gather(GraphT<EdgeWeightType> &graph, VertexFrontier &frontier, unsigned int cta_id, unsigned int num_cta) {
        int32_t thread_id = threadIdx.x + blockDim.x * cta_id;
        int32_t tot_size = AccessorType::getSize(frontier);
	int32_t idx;
	if(thread_id < tot_size) {
		idx = AccessorType::getElement(frontier, thread_id);
		graph.strict_sum[thread_id] = graph.d_get_degree(idx);
	}
}

template <typename AccessorType, typename EdgeWeightType>
void __global__ strict_gather_kernel(GraphT<EdgeWeightType> graph, VertexFrontier frontier) {
	strict_gather<AccessorType, EdgeWeightType>(graph, frontier, blockIdx.x, gridDim.x);
}
void __device__ strict_get_partial_sum(int32_t *elt, int32_t *buf, int32_t f_size, int32_t nnz_per_blk, unsigned int cta_id, unsigned int num_cta)
{
	int32_t idx = cta_id*nnz_per_blk + threadIdx.x;
	int32_t upper_idx = (cta_id+1)*nnz_per_blk;
	if(upper_idx > f_size) upper_idx = f_size;
	int32_t accum=0;

	__shared__ int32_t sm_accum[32];
	for(int32_t i=idx; i<upper_idx; i+=blockDim.x) {
		accum += elt[i];
	}
	accum += __shfl_down_sync((uint32_t)-1, accum, 16);
	accum += __shfl_down_sync((uint32_t)-1, accum, 8);
	accum += __shfl_down_sync((uint32_t)-1, accum, 4);
	accum += __shfl_down_sync((uint32_t)-1, accum, 2);
	accum += __shfl_down_sync((uint32_t)-1, accum, 1);
	if(threadIdx.x % 32 == 0) {
		sm_accum[threadIdx.x/32] = accum;
	}
	__syncthreads();
	if(threadIdx.x < PREFIX_BLK/32) {
		accum = sm_accum[threadIdx.x];
	} else {
		accum = 0;
	}
	__syncwarp();
	if(threadIdx.x < 32) {
		accum += __shfl_down_sync((uint32_t)-1, accum, 16);
		accum += __shfl_down_sync((uint32_t)-1, accum, 8);
		accum += __shfl_down_sync((uint32_t)-1, accum, 4);
		accum += __shfl_down_sync((uint32_t)-1, accum, 2);
		accum += __shfl_down_sync((uint32_t)-1, accum, 1);
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		buf[cta_id] = accum;
	}
}
void __global__ strict_get_partial_sum_kernel(int32_t *elt, int32_t *buf, int32_t f_size, int32_t nnz_per_blk) {
	strict_get_partial_sum(elt, buf, f_size, nnz_per_blk, blockIdx.x, gridDim.x);
}

void __device__ strict_local_prefix_sum(int32_t *elt, int32_t *buf, int32_t *glt, int32_t prefix_mode, int32_t f_size, int32_t nnz_per_blk, unsigned int cta_id, unsigned int num_cta) {
	__shared__ int32_t sm_deg[PREFIX_BLK];

	int32_t lane = (threadIdx.x&31);

	// prefix sum
	int32_t cosize = blockDim.x;
	int32_t tot_deg;
	int32_t phase = threadIdx.x;
	int32_t off=32;

	int32_t base_offset = 0;
	if(cta_id > 0) base_offset = buf[cta_id];

	int32_t idx = cta_id*nnz_per_blk + threadIdx.x;
	int32_t upper_idx = (cta_id+1)*nnz_per_blk;
	if(upper_idx > f_size) upper_idx = f_size;

	for(int32_t i=idx; i<(cta_id+1)*nnz_per_blk; i += blockDim.x) {
		int32_t deg = 0;
		if(i < upper_idx) deg = elt[i];

		for(int32_t d=2; d<=32; d<<=1) {
			int32_t temp = __shfl_up_sync((uint32_t)-1, deg, d/2);
			if (lane % d == d - 1) deg += temp;
		}
		sm_deg[threadIdx.x] = deg;

		for(int32_t d=cosize>>(1+5); d>0; d>>=1){
			__syncthreads();
			if(phase<d){
				int32_t ai = off*(2*phase+1)-1;
				int32_t bi = off*(2*phase+2)-1;
				sm_deg[bi] += sm_deg[ai];
			}
			off<<=1;
		}

		__syncthreads();
		tot_deg = sm_deg[cosize-1];
		__syncthreads();
		if(!phase) sm_deg[cosize-1]=0;
		__syncthreads();

		for(int32_t d=1; d<(cosize>>5); d<<=1){
			off >>=1;
			__syncthreads();
			if(phase<d){
				int32_t ai = off*(2*phase+1)-1;
				int32_t bi = off*(2*phase+2)-1;

				int32_t t = sm_deg[ai];
				sm_deg[ai]  = sm_deg[bi];
				sm_deg[bi] += t;
			}
		}
		__syncthreads();
		deg = sm_deg[threadIdx.x];
		__syncthreads();
		for(int32_t d=32; d>1; d>>=1) {
			int32_t temp_big = __shfl_down_sync((uint32_t)-1, deg, d/2);
			int32_t temp_small = __shfl_up_sync((uint32_t)-1, deg, d/2);
			if (lane % d == d/2 - 1) deg = temp_big;
			else if(lane % d == d - 1) deg += temp_small;
		}
		//sm_deg[threadIdx.x] = deg;
		if(i < upper_idx) {
			elt[i] = base_offset + deg;
		}
		__syncthreads();
		base_offset += tot_deg;

	}
	__syncthreads();
	if (prefix_mode == 1 && threadIdx.x == 0) {
		glt[0] = base_offset;
	}
}
void __global__ strict_local_prefix_sum_kernel(int32_t *elt, int32_t *buf, int32_t *glt, int32_t prefix_mode, int32_t f_size, int32_t nnz_per_blk) {
	strict_local_prefix_sum(elt, buf, glt, prefix_mode, f_size, nnz_per_blk, blockIdx.x, gridDim.x);
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __device__ strict_load_balance(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier, unsigned int cta_id, unsigned int num_cta) {

	__shared__ int32_t sm_idx[STRICT_SM_SIZE], sm_deg[STRICT_SM_SIZE], sm_loc[STRICT_SM_SIZE];
	//int32_t thread_id = threadIdx.x + blockDim.x * cta_id;
	int32_t tot_size = AccessorType::getSize(input_frontier);

        int32_t index, src_idx;
	//int32_t deg;

	// if(cta_id == num_cta - 1) return;
	// can be fused
	//bool last_tb = (cta_id == (graph.strict_grid_sum[0] + NNZ_PER_BLOCK-1)/NNZ_PER_BLOCK-1);
	int32_t start_row = binary_search_upperbound(&graph.strict_sum[0], tot_size, NNZ_PER_BLOCK*cta_id)-1;
	int32_t end_row = binary_search_upperbound(&graph.strict_sum[0], tot_size, NNZ_PER_BLOCK*(cta_id+1))-1;

	int32_t row_size = end_row - start_row + 1;
	//int32_t start_idx;

	//if(row_size <= STRICT_SM_SIZE) {
	if(row_size <= -1 ) {
		if(threadIdx.x < row_size) {
			index = AccessorType::getElement(input_frontier, start_row+threadIdx.x);
			//deg = graph.d_get_degree(index);

			sm_idx[threadIdx.x] = index;
			int32_t tmp_deg = graph.strict_sum[start_row + threadIdx.x] - cta_id * NNZ_PER_BLOCK;
			if(tmp_deg >= 0) {
				sm_deg[threadIdx.x] = tmp_deg;
				sm_loc[threadIdx.x] = graph.d_src_offsets[index];
			} else {
				sm_deg[threadIdx.x] = 0;
				sm_loc[threadIdx.x] = graph.d_src_offsets[index] - tmp_deg;
			}
		} else {
			//deg = 0;
			sm_deg[threadIdx.x] = INT_MAX;
		}
		__syncthreads();

		//int32_t lane = (threadIdx.x&31);
		int32_t offset = 0;


		int32_t tot_deg = graph.strict_grid_sum[0] - cta_id * NNZ_PER_BLOCK;
		if(tot_deg > NNZ_PER_BLOCK) tot_deg = NNZ_PER_BLOCK;
		//int32_t tot_deg;
		//if(!last_tb) tot_deg = NNZ_PER_BLOCK;
		//else tot_deg = (graph.strict_grid_sum[0] - 1) % NNZ_PER_BLOCK + 1;

		//int32_t phase = threadIdx.x;
		//int32_t off=32;

		int32_t width = row_size;
		for(int32_t i=threadIdx.x; i<tot_deg; i+=blockDim.x) {
			int32_t id = binary_search_upperbound(&sm_deg[offset], width, i)-1;
			if(id >= width) continue;
			src_idx = sm_idx[offset + id];
			if (src_filter(src_idx) == false)
				continue;
			int32_t ei = sm_loc[offset + id] + i - sm_deg[offset + id];
			if(ei >= graph.num_edges) break;
			int32_t dst_idx = graph.d_edge_dst[ei];
			load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
		}
	} else {
		int32_t tot_deg = graph.strict_grid_sum[0] - cta_id * NNZ_PER_BLOCK;
		if(tot_deg > NNZ_PER_BLOCK) tot_deg = NNZ_PER_BLOCK;
		//if(!last_tb) tot_deg = NNZ_PER_BLOCK;
		//else tot_deg = (graph.strict_grid_sum[0] - 1) % NNZ_PER_BLOCK + 1;

		int32_t width = row_size;
		//int32_t offset = 0;

		for(int32_t i=cta_id*NNZ_PER_BLOCK+threadIdx.x; i<cta_id*NNZ_PER_BLOCK+tot_deg; i+=blockDim.x) {
			int32_t id = binary_search_upperbound(&graph.strict_sum[start_row], width, i)-1;
			if(id >= width) continue;
			src_idx = AccessorType::getElement(input_frontier, start_row+id);
			if (src_filter(src_idx) == false)
				continue;
			int32_t ei = graph.d_src_offsets[src_idx] + i - graph.strict_sum[start_row + id];
			if(ei >= graph.num_edges) break;
			int32_t dst_idx = graph.d_edge_dst[ei];
			load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
		}


	}
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)>
void __global__ strict_load_balance_kernel(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	strict_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, blockIdx.x, gridDim.x);
}

template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __host__ strict_load_balance_host(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int num_threads = AccessorType::getSizeHost(input_frontier);	
	int num_cta = (num_threads + CTA_SIZE - 1)/CTA_SIZE;
	int cta_size = CTA_SIZE;	
	strict_gather_kernel<AccessorType, EdgeWeightType><<<num_cta, cta_size>>>(graph, input_frontier);
	
	int32_t tot_blk = NUM_CTA;	
	int32_t low_blk = (num_threads + PREFIX_BLK - 1)/PREFIX_BLK;
	if (low_blk < tot_blk)
		tot_blk = low_blk;	
	
	int32_t gran = PREFIX_BLK * tot_blk;
	int32_t nnz_per_thread = (num_threads + gran - 1)/gran;
	int32_t nnz_per_blk = (nnz_per_thread * PREFIX_BLK);


	strict_get_partial_sum_kernel<<<tot_blk, PREFIX_BLK>>>(graph.strict_sum, graph.strict_cta_sum, num_threads, nnz_per_blk);
	
	strict_local_prefix_sum_kernel<<<1, PREFIX_BLK>>>(graph.strict_cta_sum, graph.strict_cta_sum, graph.strict_grid_sum, 1, tot_blk + 1, (tot_blk + PREFIX_BLK)/PREFIX_BLK * PREFIX_BLK);
	strict_local_prefix_sum_kernel<<<tot_blk, PREFIX_BLK>>>(graph.strict_sum, graph.strict_cta_sum, graph.strict_grid_sum, 0, num_threads, nnz_per_blk);
	cudaMemcpy(&num_threads, graph.strict_grid_sum, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaCheckLastError();
	num_cta = (num_threads + CTA_SIZE - 1)/CTA_SIZE;
	cta_size = CTA_SIZE;	

	strict_load_balance_kernel<EdgeWeightType, load_balance_payload, AccessorType, src_filter><<<num_cta, cta_size>>>(graph, input_frontier, output_frontier);	
}
template <typename EdgeWeightType, load_balance_payload_type<EdgeWeightType> load_balance_payload, typename AccessorType, bool src_filter(int32_t)> 
void __device__ strict_load_balance_device(GraphT<EdgeWeightType> &graph, VertexFrontier &input_frontier, VertexFrontier &output_frontier) {
	int num_threads = AccessorType::getSize(input_frontier);	
	int num_cta = (num_threads + CTA_SIZE - 1)/CTA_SIZE;

	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {	
		strict_gather<AccessorType, EdgeWeightType>(graph, input_frontier, cta_id, num_cta);
		__syncthreads();
	}
	this_grid().sync();
	
	int32_t tot_blk = NUM_CTA;	
	int32_t low_blk = (num_threads + PREFIX_BLK - 1)/PREFIX_BLK;
	if (low_blk < tot_blk)
		tot_blk = low_blk;	
	int32_t gran = PREFIX_BLK * tot_blk;
	int32_t nnz_per_thread = (num_threads + gran - 1)/gran;
	int32_t nnz_per_blk = (nnz_per_thread * PREFIX_BLK);

	for (int32_t cta_id = blockIdx.x; cta_id < tot_blk; cta_id += gridDim.x) {	
		strict_get_partial_sum(graph.strict_sum, graph.strict_cta_sum, num_threads, nnz_per_blk, cta_id, tot_blk);
		__syncthreads();
	}
	this_grid().sync();
	if (blockIdx.x == 0) {
		strict_local_prefix_sum(graph.strict_cta_sum, graph.strict_cta_sum, graph.strict_grid_sum, 1, tot_blk + 1, (tot_blk + PREFIX_BLK)/PREFIX_BLK * PREFIX_BLK, blockIdx.x, 1);
	}	
	this_grid().sync();
	for (int32_t cta_id = blockIdx.x; cta_id < tot_blk; cta_id += gridDim.x) {	
		strict_local_prefix_sum(graph.strict_sum, graph.strict_cta_sum, graph.strict_grid_sum, 0, num_threads, nnz_per_blk, cta_id, tot_blk);
		__syncthreads();
	}
	this_grid().sync();
	num_threads = graph.strict_grid_sum[0];
	num_cta = (num_threads + CTA_SIZE - 1)/CTA_SIZE;
	for (int32_t cta_id = blockIdx.x; cta_id < num_cta; cta_id += gridDim.x) {	
		strict_load_balance<EdgeWeightType, load_balance_payload, AccessorType, src_filter>(graph, input_frontier, output_frontier, cta_id, num_cta);
		__syncthreads();
	}
	this_grid().sync();
	
}

}
#endif
