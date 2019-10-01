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

	uint32_t* d_bit_map_input;
	uint32_t* d_bit_map_output;

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
static int32_t builtin_getVertexSetSize(VertexFrontier &frontier) {
	int32_t curr_size = 0;
	cudaMemcpy(&curr_size, frontier.d_num_elems_input, sizeof(int32_t), cudaMemcpyDeviceToHost);
	return curr_size;	
}
class AccessorSparse {
public:
	static int32_t __device__ getSize(VertexFrontier &frontier) {
		return frontier.d_num_elems_input[0];
	}
	static int32_t __device__ getElement(VertexFrontier &frontier, int32_t index) {
		return frontier.d_sparse_queue_input[index];
	}
	static int32_t getSizeHost(VertexFrontier &frontier) {
		return builtin_getVertexSetSize(frontier);
	}
};
class AccessorAll {
public:
	static int32_t __device__ getSize(VertexFrontier &frontier) {
		return frontier.max_num_elems;
	}
	static int32_t __device__ getElement(VertexFrontier &frontier, int32_t index) {
		return index;
	}
	static int32_t getSizeHost(VertexFrontier &frontier) {
		return frontier.max_num_elems;
	}
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
	
	int32_t num_byte_for_bitmap = (num_vertices + sizeof(uint32_t) - 1)/sizeof(uint32_t);
	cudaMalloc(&frontier.d_bit_map_input, sizeof(uint32_t) * num_byte_for_bitmap);
	cudaMalloc(&frontier.d_bit_map_output, sizeof(uint32_t) * num_byte_for_bitmap);
	
	cudaMemset(frontier.d_bit_map_input, 0, sizeof(uint32_t) * num_byte_for_bitmap);	
	cudaMemset(frontier.d_bit_map_output, 0, sizeof(uint32_t) * num_byte_for_bitmap);	
	cudaCheckLastError();

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
	if (byte_map[vertex_id] == 1)
		return;
	byte_map[vertex_id] = 1;
	atomicAggInc(byte_map_size);
}
static bool __device__ checkBit(uint32_t* array, int32_t index) {	
	uint32_t * address = array + index / sizeof(uint32_t);
	return (*address & (1 << (index % sizeof(uint32_t))));
}
static bool __device__ setBit(uint32_t* array, int32_t index) {
	uint32_t * address = array + index / sizeof(uint32_t);	
	return atomicOr(address, (1 << (index % sizeof(uint32_t)))) & (1 << (index % sizeof(uint32_t)));
}
static void __device__ enqueueVertexBitmap(uint32_t* bit_map, int32_t * bit_map_size, int32_t vertex_id) {
	// We need atomics here because of bit manipulations
	if (checkBit(bit_map, vertex_id)) 
		return;
	if (!setBit(bit_map, vertex_id))
		atomicAggInc(bit_map_size);	
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
static void swap_bitmaps(VertexFrontier &frontier) {
	int32_t *temp = frontier.d_num_elems_input;
	frontier.d_num_elems_input = frontier.d_num_elems_output;
	frontier.d_num_elems_output = temp;
	
	uint32_t* temp2;
	temp2 = frontier.d_bit_map_input;
	frontier.d_bit_map_input = frontier.d_bit_map_output;
	frontier.d_bit_map_output = temp2;

	cudaMemset(frontier.d_num_elems_output, 0, sizeof(int32_t));		
	int32_t num_byte_for_bitmap = (frontier.max_num_elems + sizeof(uint32_t) - 1)/sizeof(uint32_t);
	cudaMemset(frontier.d_bit_map_output, 0, sizeof(uint32_t) * num_byte_for_bitmap);
	cudaCheckLastError();
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
static void __global__ prepare_sparse_from_bitmap(VertexFrontier frontier) {
	for (int32_t node_id = blockDim.x * blockIdx.x + threadIdx.x; node_id < frontier.max_num_elems; node_id += blockDim.x * gridDim.x) {
		if (checkBit(frontier.d_bit_map_input, node_id)) {
			enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, node_id);
		}
	}
}

static void __global__ prepare_bytemap_from_sparse(VertexFrontier frontier) {
	for (int32_t node_idx = blockDim.x * blockIdx.x + threadIdx.x; node_idx < frontier.d_num_elems_input[0]; node_idx += blockDim.x * gridDim.x) {
		int32_t node_id = frontier.d_sparse_queue_input[node_idx];
		enqueueVertexBytemap(frontier.d_byte_map_output, frontier.d_num_elems_output, node_id);
	}
}
static void __global__ prepare_bytemap_from_bitmap(VertexFrontier frontier) {
	for (int32_t node_id = blockDim.x * blockIdx.x + threadIdx.x; node_id < frontier.max_num_elems; node_id += blockDim.x * gridDim.x) {
		if (checkBit(frontier.d_bit_map_input, node_id)) {
			enqueueVertexBytemap(frontier.d_byte_map_output, frontier.d_num_elems_output, node_id);
		}
	}
}

static void __global__ prepare_bitmap_from_sparse(VertexFrontier frontier) {
	for (int32_t node_idx = blockDim.x * blockIdx.x + threadIdx.x; node_idx < frontier.d_num_elems_input[0]; node_idx += blockDim.x * gridDim.x) {
		int32_t node_id = frontier.d_sparse_queue_input[node_idx];
		enqueueVertexBitmap(frontier.d_bit_map_output, frontier.d_num_elems_output, node_id);
	}
}
static void __global__ prepare_bitmap_from_bytemap(VertexFrontier frontier) {
	for (int32_t node_id = blockDim.x * blockIdx.x + threadIdx.x; node_id < frontier.max_num_elems; node_id += blockDim.x * gridDim.x) {
		if (frontier.d_byte_map_input[node_id] == 1) {
			enqueueVertexBitmap(frontier.d_bit_map_output, frontier.d_num_elems_output, node_id);
		}
	}
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
static void vertex_set_prepare_boolmap(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		prepare_bytemap_from_sparse<<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bytemaps(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		return;
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		prepare_bytemap_from_bitmap<<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bytemaps(frontier);
		return;
	}
}
static void vertex_set_prepare_bitmap(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		prepare_bitmap_from_sparse<<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bitmaps(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		prepare_bitmap_from_bytemap<<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bitmaps(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		return;
	}
}
bool __device__ true_function(int32_t _) {
	return true;
}
template <bool to_func(int32_t)>
static void __global__ vertex_set_create_reverse_sparse_queue_kernel(VertexFrontier frontier) {
	for (int32_t node_id = blockDim.x * blockIdx.x + threadIdx.x; node_id < frontier.max_num_elems; node_id += blockDim.x * gridDim.x) {
		if ((to_func(node_id)))
			enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, node_id);
	}	
}

template <bool to_func(int32_t)>
static void vertex_set_create_reverse_sparse_queue(VertexFrontier &frontier) {
	vertex_set_create_reverse_sparse_queue_kernel<to_func><<<NUM_CTA, CTA_SIZE>>>(frontier);
	swap_queues(frontier);	
}

}

#endif

