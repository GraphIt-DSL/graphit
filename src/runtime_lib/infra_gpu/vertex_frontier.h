#ifndef GPU_VERTEX_FRONTIER_H
#define GPU_VERTEX_FRONTIER_H

#include "infra_gpu/support.h"
namespace gpu_runtime {
struct VertexFrontier {
	int32_t max_num_elems; 

	int32_t *d_num_elems_input;
	int32_t *d_num_elems_output;

	int32_t * d_sparse_queue_input;
	int32_t * d_sparse_queue_output;
	
	unsigned char* d_byte_map_input;
	unsigned char* d_byte_map_output;

	unsigned char* d_bit_map_input;
	unsigned char* d_bit_map_output;

	int32_t *d_dedup_counters;
	int32_t curr_dedup_counter;

	// Extend this to check the current representation
	enum format_ready_type {
		SPARSE,
		BITMAP,
		BYTEMAP		
	};

	format_ready_type format_ready;
};
static VertexFrontier create_new_vertex_set(int32_t num_vertices) {
	VertexFrontier frontier;
	cudaMalloc(&frontier.d_num_elems_input, sizeof(int32_t));
	cudaMalloc(&frontier.d_num_elems_output, sizeof(int32_t));
	cudaMemset(frontier.d_num_elems_input, 0, sizeof(int32_t));
	cudaMemset(frontier.d_num_elems_output, 0, sizeof(int32_t));

	cudaMalloc(&frontier.d_sparse_queue_input, sizeof(int32_t) * num_vertices * 6);
	cudaMalloc(&frontier.d_sparse_queue_output, sizeof(int32_t) * num_vertices * 6);

	cudaMalloc(&frontier.d_byte_map_input, sizeof(unsigned char) * num_vertices);
	cudaMalloc(&frontier.d_byte_map_output, sizeof(unsigned char) * num_vertices);
	
	cudaMemset(frontier.d_byte_map_input, 0, sizeof(unsigned char) * num_vertices);
	cudaMemset(frontier.d_byte_map_output, 0, sizeof(unsigned char) * num_vertices);
	
	int32_t num_byte_for_bitmap = (num_vertices + 7)/8;
	cudaMalloc(&frontier.d_bit_map_input, sizeof(unsigned char) * num_byte_for_bitmap);
	cudaMalloc(&frontier.d_bit_map_output, sizeof(unsigned char) * num_byte_for_bitmap);
	
	cudaMemset(frontier.d_bit_map_input, 0, sizeof(unsigned char) * num_byte_for_bitmap);	
	cudaMemset(frontier.d_bit_map_output, 0, sizeof(unsigned char) * num_byte_for_bitmap);	

	frontier.max_num_elems = num_vertices;

	frontier.curr_dedup_counter = 0;
	cudaMalloc(&frontier.d_dedup_counters, sizeof(int32_t) * num_vertices);
	cudaMemset(frontier.d_dedup_counters, 0, sizeof(int32_t) * num_vertices);

	frontier.format_ready = VertexFrontier::SPARSE;

	cudaCheckLastError();

	return frontier;
}

static void builtin_addVertex(VertexFrontier &frontier, int32_t vid) {
	int32_t curr_size;
	cudaMemcpy(&curr_size, frontier.d_num_elems_input, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(frontier.d_sparse_queue_input + curr_size, &vid, sizeof(int32_t), cudaMemcpyHostToDevice);
	curr_size++;
	
	cudaMemcpy(frontier.d_num_elems_input, &curr_size, sizeof(int32_t), cudaMemcpyHostToDevice);
}
static void __device__ enqueueVertexSparseQueue(int32_t *sparse_queue, int32_t *sparse_queue_size, int32_t vertex_id) {
	// Simple enqueuVertex implementation 
	// Each thread adds on it's own
	// TODO: Optimize with warp reduce

	//int32_t pos = atomicAdd(sparse_queue_size, 1);
	int32_t pos = atomicAggInc(sparse_queue_size);
	sparse_queue[pos] = vertex_id;
	
}
static void __device__ enqueueVertexBytemap(unsigned char* byte_map, int32_t *byte_map_size, int32_t vertex_id) {
	// We are not using atomic operation here because races are benign here
	byte_map[vertex_id] = 1;
	atomicAggInc(byte_map_size);
}
static int32_t builtin_getVertexSetSize(VertexFrontier &frontier) {
	int32_t curr_size = 0;
	cudaMemcpy(&curr_size, frontier.d_num_elems_input, sizeof(int32_t), cudaMemcpyDeviceToHost);
	return curr_size;	
}
static void swap_queues(VertexFrontier &frontier) {
	int32_t *temp = frontier.d_num_elems_input;
	frontier.d_num_elems_input = frontier.d_num_elems_output;
	frontier.d_num_elems_output = temp;
	
	temp = frontier.d_sparse_queue_input;
	frontier.d_sparse_queue_input = frontier.d_sparse_queue_output;
	frontier.d_sparse_queue_output = temp;

	cudaMemset(frontier.d_num_elems_output, 0, sizeof(int32_t));	
}
static void swap_bytemaps(VertexFrontier &frontier) {
	int32_t *temp = frontier.d_num_elems_input;
	frontier.d_num_elems_input = frontier.d_num_elems_output;
	frontier.d_num_elems_output = temp;
	
	unsigned char* temp2;
	temp2 = frontier.d_byte_map_input;
	frontier.d_byte_map_input = frontier.d_byte_map_output;
	frontier.d_byte_map_output = temp2;

	cudaMemset(frontier.d_num_elems_output, 0, sizeof(int32_t));	
	cudaMemset(frontier.d_byte_map_output, 0, sizeof(unsigned char) * frontier.max_num_elems);
}
static void __device__ dedup_frontier_device(VertexFrontier &frontier) {
	for(int32_t vidx = threadIdx.x + blockDim.x * blockIdx.x; vidx < frontier.d_num_elems_input[0]; vidx += blockDim.x * gridDim.x) {
		int32_t vid = frontier.d_sparse_queue_input[vidx];
		if (frontier.d_dedup_counters[vid] < frontier.curr_dedup_counter) {
			enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, vid);
			frontier.d_dedup_counters[vid] = frontier.curr_dedup_counter;	
		}
	}
}
static void __global__ dedup_frontier_kernel(VertexFrontier frontier) {
	dedup_frontier_device(frontier);	
}
static void dedup_frontier(VertexFrontier &frontier) {
	frontier.curr_dedup_counter++;
	dedup_frontier_kernel<<<NUM_CTA, CTA_SIZE>>>(frontier);
	swap_queues(frontier);
}

static void __global__ prepare_sparse_from_bytemap(VertexFrontier frontier) {
	for (int32_t node_id = blockDim.x * blockIdx.x + threadIdx.x; node_id < frontier.max_num_elems; node_id += blockDim.x * gridDim.x) {
		if (frontier.d_byte_map_input[node_id] == 1) {
			enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, node_id);
		}
	}
}
static void __global__ prepare_sparse_from_bitmap(VertexFrontier &frontier) {
}

static void vertex_set_prepare_sparse(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE)
		return;
	else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		prepare_sparse_from_bytemap<<<NUM_CTA, CTA_SIZE>>>(frontier);	
		swap_queues(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		prepare_sparse_from_bitmap<<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_queues(frontier);	
		return;	
	}	
}
}

#endif

