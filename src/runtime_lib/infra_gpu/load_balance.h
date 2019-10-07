#ifndef GRAPHIT_GPU_LOAD_BALANCE_H
#define GRAPHIT_GPU_LOAD_BALANCE_H

#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"
#include "infra_gpu/support.h"


namespace gpu_runtime {

__device__ inline int32_t upperbound(int32_t *array, int32_t len, int32_t key){
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

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
void __device__ vertex_based_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	int32_t vid = threadIdx.x + blockDim.x * blockIdx.x;
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


template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
void __device__ warp_based_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {

	__shared__ int32_t sm_idx[CTA_SIZE], sm_deg[CTA_SIZE], sm_loc[CTA_SIZE];
	int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t tot_size = AccessorType::getSize(input_frontier);

        int32_t deg, index, index_size, src_idx;
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
                int32_t temp = __shfl_up_sync(-1, deg, d);
                if (lane >= d) deg += temp;
        }
        int32_t tot_deg = __shfl_sync(-1, deg, 31);
        if(lane == 31) deg = 0;
        sm_deg[offset + ((lane+1)&31)] = deg;
        __syncthreads();

        // compute
        int32_t width = thread_id - lane + 32;
        if(tot_size < width) width = tot_size;
        width -= thread_id - lane;

        for(int32_t i=lane; i<tot_deg; i+=32) {
                int32_t id = upperbound(&sm_deg[offset], width, i)-1;

                src_idx = sm_idx[offset + id];

                int32_t ei = sm_loc[offset + id] + i - sm_deg[offset + id];
                int32_t dst_idx = graph.d_edge_dst[ei];
		load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
        }
}

template <typename AccessorType>
void __host__ warp_based_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
void __device__ tb_based_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {

	__shared__ int32_t sm_idx[CTA_SIZE], sm_deg[CTA_SIZE], sm_loc[CTA_SIZE];
	int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t tot_size = AccessorType::getSize(input_frontier);

        int32_t deg, index, index_size, src_idx;
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

        int32_t lane = (threadIdx.x&31);
        int32_t offset = 0;
	
	// prefix sum
	int32_t cosize = blockDim.x;
	int32_t tot_deg;
	int32_t phase = threadIdx.x;
	int32_t off=32;

	for(int32_t d=2; d<=32; d<<=1) {
		int32_t temp = __shfl_up_sync(-1, deg, d/2);
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
		int32_t temp_big = __shfl_down_sync(-1, deg, d/2);
		int32_t temp_small = __shfl_up_sync(-1, deg, d/2);
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
                int32_t id = upperbound(&sm_deg[offset], width, i)-1;

                if(id >= width) continue;
                src_idx = sm_idx[offset + id];

                int32_t ei = sm_loc[offset + id] + i - sm_deg[offset + id];
                int32_t dst_idx = graph.d_edge_dst[ei];
		load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
        }
}



template <typename AccessorType>
void __host__ tb_based_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

#define NNZ_PER_BLOCK (CTA_SIZE)
#define STRICT_SM_SIZE (CTA_SIZE)



template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
void __device__ STRICT_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {

	__shared__ int32_t sm_idx[STRICT_SM_SIZE], sm_deg[STRICT_SM_SIZE], sm_loc[STRICT_SM_SIZE];
	int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t tot_size = AccessorType::getSize(input_frontier);

        int32_t deg, index, index_size, src_idx;

	// can be fused
	bool last_tb = (blockIdx.x == (input_frontier.d_sparse_queue_input[graph.num_vertices+tot_size]+NNZ_PER_BLOCK-1)/NNZ_PER_BLOCK-1);
	int32_t start_row = upperbound(&input_frontier.d_sparse_queue_input[graph.num_vertices], tot_size, NNZ_PER_BLOCK*blockIdx.x)-1;
	int32_t end_row = upperbound(&input_frontier.d_sparse_queue_input[graph.num_vertices], tot_size, NNZ_PER_BLOCK*(blockIdx.x+1))-1;

	int32_t row_size = end_row - start_row + 1;
	int32_t start_idx;

	// row_size <= STRICT_SM_SIZE 
	if(threadIdx.x < row_size) {
		index = AccessorType::getElement(input_frontier, start_row+threadIdx.x);
		deg = graph.d_get_degree(index);

		sm_idx[threadIdx.x] = index;
		int32_t tmp_deg = input_frontier.d_sparse_queue_input[graph.num_vertices + start_row + threadIdx.x] - blockIdx.x * NNZ_PER_BLOCK;
		if(tmp_deg >= 0) {
			sm_deg[threadIdx.x] = tmp_deg;
			sm_loc[threadIdx.x] = graph.d_src_offsets[index];
		} else {
			sm_deg[threadIdx.x] = 0;
			sm_loc[threadIdx.x] = graph.d_src_offsets[index] - tmp_deg;
		}
	} else {
		deg = 0;
		sm_deg[threadIdx.x] = 1073742418;
	}
	__syncthreads();

	int32_t lane = (threadIdx.x&31);
	int32_t offset = 0;
	
	int32_t tot_deg;
	if(!last_tb) tot_deg = NNZ_PER_BLOCK;
	else tot_deg = (input_frontier.d_sparse_queue_input[graph.num_vertices + tot_size] - 1) % NNZ_PER_BLOCK + 1;

	int32_t phase = threadIdx.x;
	int32_t off=32;

	int32_t width = row_size;
	for(int32_t i=threadIdx.x; i<tot_deg; i+=blockDim.x) {
		int32_t id = upperbound(&sm_deg[offset], width, i)-1;
		if(id >= width) continue;
		src_idx = sm_idx[offset + id];
		int32_t ei = sm_loc[offset + id] + i - sm_deg[offset + id];
		int32_t dst_idx = graph.d_edge_dst[ei];
		load_balance_payload(graph, src_idx, dst_idx, ei, input_frontier, output_frontier);
	}
}

template <typename AccessorType, typename EdgeWeightType>
void __global__ STRICT_gather(GraphT<EdgeWeightType> g, VertexFrontier frontier)
{
        int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
        int32_t tot_size = AccessorType::getSize(frontier);
	int32_t idx, deg;
	if(thread_id < tot_size) {
		idx = AccessorType::getElement(frontier, thread_id);
		frontier.d_sparse_queue_input[thread_id+g.num_vertices] = g.d_get_degree(idx);
	}
}


template <typename AccessorType>
void __host__ STRICT_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

#define MID_BIN (32)
#define LARGE_BIN (CTA_SIZE)

template <typename AccessorType, typename EdgeWeightType>
void __global__ split_frontier(GraphT<EdgeWeightType> g, VertexFrontier frontier)
{
        int32_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
        int32_t tot_size = AccessorType::getSize(frontier);
	int32_t idx, deg;
	if(thread_id < tot_size) {
		idx = AccessorType::getElement(frontier, thread_id);
		deg = g.d_get_degree(idx);
		if(deg <= MID_BIN) {
			int32_t k = atomicAggInc(&frontier.d_num_elems_output[0]);
			frontier.d_sparse_queue_output[k] = idx;
		} else if(deg <= LARGE_BIN) {
			int32_t k = atomicAggInc(&frontier.d_num_elems_output[1]);
			frontier.d_sparse_queue_output[k+g.num_vertices] = idx;
		} else {
			int32_t k = atomicAggInc(&frontier.d_num_elems_output[2]);
			frontier.d_sparse_queue_output[k+g.num_vertices*2] = idx;
		}
	}

}

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
void __device__ TWC_load_balance_mid(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	int32_t vid = (threadIdx.x + blockDim.x * blockIdx.x)/MID_BIN;
	if (vid >= input_frontier.d_num_elems_input[1])
		return;

	int32_t src = input_frontier.d_sparse_queue_input[vid+graph.num_vertices];
	for (int32_t eid = graph.d_src_offsets[src]+(threadIdx.x%MID_BIN); eid < graph.d_src_offsets[src+1]; eid+=MID_BIN) {
		if (src_filter(src) == false)
			break;
		int32_t dst = graph.d_edge_dst[eid];
		load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);
	}
}

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
void __device__ TWC_load_balance_large(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	int32_t vid = (threadIdx.x + blockDim.x * blockIdx.x)/LARGE_BIN;
	if (vid >= input_frontier.d_num_elems_input[2])
		return;
	int32_t src = input_frontier.d_sparse_queue_input[vid+graph.num_vertices*2];
	for (int32_t eid = graph.d_src_offsets[src]+(threadIdx.x%LARGE_BIN); eid < graph.d_src_offsets[src+1]; eid+=LARGE_BIN) {
		if (src_filter(src) == false)
			break;
		int32_t dst = graph.d_edge_dst[eid];
		load_balance_payload(graph, src, dst, eid, input_frontier, output_frontier);
	}
}

template <typename AccessorType>
void __host__ TWC_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = AccessorType::getSizeHost(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

template <typename AccessorType>
void __host__ TWC_load_balance_info_mid_bin(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads;
       	cudaMemcpy(&num_threads, &frontier.d_num_elems_input[1], sizeof(int32_t), cudaMemcpyDeviceToHost);
	num_cta = (num_threads*MID_BIN + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}

template <typename AccessorType>
void __host__ TWC_load_balance_info_large_bin(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads;
       	cudaMemcpy(&num_threads, &frontier.d_num_elems_input[2], sizeof(int32_t), cudaMemcpyDeviceToHost);
	num_cta = (num_threads*LARGE_BIN + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}


#define STAGE_1_SIZE (8)
#define WARP_SIZE (32)
template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier, VertexFrontier), typename AccessorType, bool src_filter(int32_t)>
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

}

#endif
