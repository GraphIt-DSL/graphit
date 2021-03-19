#ifndef GPU_VERTEX_FRONTIER_H
#define GPU_VERTEX_FRONTIER_H

#include "infra_gpu/support.h"
#include <cooperative_groups.h>
#ifndef FRONTIER_MULTIPLIER
#define FRONTIER_MULTIPLIER (6)
#endif
using namespace cooperative_groups;
namespace gpu_runtime {
class VertexFrontier {

 public:
  
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
	
	// PriorityQueue related trackers
	int32_t* d_priority_array;
	int32_t priority_cutoff;
};


static void cudaFreeSafe(void* ptr) {
	cudaFree(ptr);
}
void delete_vertex_frontier(VertexFrontier &frontier) {
	cudaFreeSafe(frontier.d_sparse_queue_input);	
	cudaFreeSafe(frontier.d_sparse_queue_output);
	cudaFreeSafe(frontier.d_num_elems_input);
	cudaFreeSafe(frontier.d_num_elems_output);
	cudaFreeSafe(frontier.d_byte_map_input);
	cudaFreeSafe(frontier.d_byte_map_output);
	cudaFreeSafe(frontier.d_bit_map_input);
	cudaFreeSafe(frontier.d_bit_map_output);
	cudaFreeSafe(frontier.d_dedup_counters);
	return;
}
static VertexFrontier sentinel_frontier;
static __device__ VertexFrontier device_sentinel_frontier;

static int32_t builtin_getVertexSetSize(VertexFrontier &frontier) {
	int32_t curr_size = 0;
	cudaMemcpy(&curr_size, frontier.d_num_elems_input, sizeof(int32_t), cudaMemcpyDeviceToHost);
	return curr_size;	
}
static int32_t __device__ device_builtin_getVertexSetSize(VertexFrontier &frontier) {
	this_grid().sync();
	return frontier.d_num_elems_input[0];
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

void __global__ initialize_frontier_all(VertexFrontier frontier) {
	for (int32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < frontier.max_num_elems; idx += blockDim.x * gridDim.x)
		frontier.d_sparse_queue_input[idx] = idx;
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
		frontier.d_num_elems_input[0] = frontier.max_num_elems;
	}
}
static VertexFrontier create_new_vertex_set(int32_t num_vertices, int32_t init_elems = 0) {
	VertexFrontier frontier;
	frontier.max_num_elems = num_vertices;
	cudaMalloc(&frontier.d_num_elems_input, sizeof(int32_t));
	cudaMalloc(&frontier.d_num_elems_output, sizeof(int32_t));
	cudaMalloc(&frontier.d_sparse_queue_input, sizeof(int32_t) * num_vertices * FRONTIER_MULTIPLIER);
	cudaMalloc(&frontier.d_sparse_queue_output, sizeof(int32_t) * num_vertices * FRONTIER_MULTIPLIER);
	
	if (num_vertices == init_elems) {
		initialize_frontier_all<<<NUM_CTA, CTA_SIZE>>>(frontier);		
	} else {
		cudaMemset(frontier.d_num_elems_input, 0, sizeof(int32_t));
	}
	cudaMemset(frontier.d_num_elems_output, 0, sizeof(int32_t));


	cudaMalloc(&frontier.d_byte_map_input, sizeof(unsigned char) * num_vertices);
	cudaMalloc(&frontier.d_byte_map_output, sizeof(unsigned char) * num_vertices);
	
	cudaMemset(frontier.d_byte_map_input, 0, sizeof(unsigned char) * num_vertices);
	cudaMemset(frontier.d_byte_map_output, 0, sizeof(unsigned char) * num_vertices);
	
	int32_t num_byte_for_bitmap = (num_vertices + sizeof(uint32_t) * 8 - 1)/(sizeof(uint32_t) * 8);
	cudaMalloc(&frontier.d_bit_map_input, sizeof(uint32_t) * num_byte_for_bitmap);
	cudaMalloc(&frontier.d_bit_map_output, sizeof(uint32_t) * num_byte_for_bitmap);
	
	cudaMemset(frontier.d_bit_map_input, 0, sizeof(uint32_t) * num_byte_for_bitmap);	
	cudaMemset(frontier.d_bit_map_output, 0, sizeof(uint32_t) * num_byte_for_bitmap);	
	cudaCheckLastError();


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
	int32_t pos = atomicAggInc(sparse_queue_size);
	sparse_queue[pos] = vertex_id;
}
static void __device__ enqueueVertexSparseQueueDedup(int32_t *sparse_queue, int32_t *sparse_queue_size, int32_t vertex_id, VertexFrontier &frontier) {
	int32_t vid = vertex_id;
	if (frontier.d_dedup_counters[vid] < frontier.curr_dedup_counter) {
		int32_t pos = atomicAggInc(sparse_queue_size);
		sparse_queue[pos] = vertex_id;
		frontier.d_dedup_counters[vid] = frontier.curr_dedup_counter;
	}
}
static void __device__ enqueueVertexSparseQueueDedupPerfect(int32_t *sparse_queue, int32_t *sparse_queue_size, int32_t vertex_id, VertexFrontier &frontier) {
	int32_t vid = vertex_id;
	if (writeMax(&frontier.d_dedup_counters[vid], frontier.curr_dedup_counter)) {
		int32_t pos = atomicAggInc(sparse_queue_size);
		sparse_queue[pos] = vertex_id;
	}
}
static void __device__ enqueueVertexBytemap(unsigned char* byte_map, int32_t *byte_map_size, int32_t vertex_id) {
	// We are not using atomic operation here because races are benign here
	if (byte_map[vertex_id] == 1)
		return;
	byte_map[vertex_id] = 1;
	atomicAggInc(byte_map_size);
}
static bool __device__ checkBit(uint32_t* array, int32_t index) {	
	uint32_t * address = array + index / (8 * sizeof(uint32_t));
	return (*address & (1 << (index % (8 * sizeof(uint32_t)))));
}
static bool __device__ setBit(uint32_t* array, int32_t index) {
	uint32_t * address = array + index / (8 * sizeof(uint32_t));
	return atomicOr(address, (1 << (index % (8 * sizeof(uint32_t))))) & (1 << (index % (8 * sizeof(uint32_t))));
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

static void __device__ swap_queues_device(VertexFrontier &frontier) {	
	int32_t *temp = frontier.d_num_elems_input;
	frontier.d_num_elems_input = frontier.d_num_elems_output;
	frontier.d_num_elems_output = temp;
	
	temp = frontier.d_sparse_queue_input;
	frontier.d_sparse_queue_input = frontier.d_sparse_queue_output;
	frontier.d_sparse_queue_output = temp;
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) 
		frontier.d_num_elems_output[0] = 0;
	this_grid().sync();
}
static void __device__ swap_queues_device_global(VertexFrontier &frontier) {	
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
		int32_t *temp = frontier.d_num_elems_input;
		frontier.d_num_elems_input = frontier.d_num_elems_output;
		frontier.d_num_elems_output = temp;
		
		temp = frontier.d_sparse_queue_input;
		frontier.d_sparse_queue_input = frontier.d_sparse_queue_output;
		frontier.d_sparse_queue_output = temp;
	}
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) 
		frontier.d_num_elems_output[0] = 0;
	this_grid().sync();
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

static void __device__ swap_bytemaps_device(VertexFrontier &frontier) {
	int32_t *temp = frontier.d_num_elems_input;
	frontier.d_num_elems_input = frontier.d_num_elems_output;
	frontier.d_num_elems_output = temp;
	
	unsigned char* temp2;
	temp2 = frontier.d_byte_map_input;
	frontier.d_byte_map_input = frontier.d_byte_map_output;
	frontier.d_byte_map_output = temp2;
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) 
		frontier.d_num_elems_output[0] = 0;
	parallel_memset(frontier.d_byte_map_output, 0, sizeof(unsigned char) * frontier.max_num_elems);		
	this_grid().sync();
}
static void __device__ swap_bytemaps_device_global(VertexFrontier &frontier) {
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
		int32_t *temp = frontier.d_num_elems_input;
		frontier.d_num_elems_input = frontier.d_num_elems_output;
		frontier.d_num_elems_output = temp;
		
		unsigned char* temp2;
		temp2 = frontier.d_byte_map_input;
		frontier.d_byte_map_input = frontier.d_byte_map_output;
		frontier.d_byte_map_output = temp2;
	}
	this_grid().sync();
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) 
		frontier.d_num_elems_output[0] = 0;
	this_grid().sync();
	parallel_memset(frontier.d_byte_map_output, 0, sizeof(unsigned char) * frontier.max_num_elems);		
	this_grid().sync();
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
	int32_t num_byte_for_bitmap = (frontier.max_num_elems + 8 * sizeof(uint32_t) - 1)/(sizeof(uint32_t) * 8);
	cudaMemset(frontier.d_bit_map_output, 0, sizeof(uint32_t) * num_byte_for_bitmap);
	cudaCheckLastError();
}
static void __device__ swap_bitmaps_device(VertexFrontier &frontier) {
	int32_t *temp = frontier.d_num_elems_input;
	frontier.d_num_elems_input = frontier.d_num_elems_output;
	frontier.d_num_elems_output = temp;
	
	uint32_t* temp2;
	temp2 = frontier.d_bit_map_input;
	frontier.d_bit_map_input = frontier.d_bit_map_output;
	frontier.d_bit_map_output = temp2;

	int32_t num_byte_for_bitmap = (frontier.max_num_elems + 8 * sizeof(uint32_t) - 1)/(sizeof(uint32_t) * 8);

	if (threadIdx.x + blockIdx.x * blockDim.x == 0) 
		frontier.d_num_elems_output[0] = 0;
	parallel_memset((unsigned char*)frontier.d_bit_map_output, 0, sizeof(uint32_t) * num_byte_for_bitmap);		
	this_grid().sync();
}
static void __device__ swap_bitmaps_device_global(VertexFrontier &frontier) {
	if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
		int32_t *temp = frontier.d_num_elems_input;
		frontier.d_num_elems_input = frontier.d_num_elems_output;
		frontier.d_num_elems_output = temp;
		
		uint32_t* temp2;
		temp2 = frontier.d_bit_map_input;
		frontier.d_bit_map_input = frontier.d_bit_map_output;
		frontier.d_bit_map_output = temp2;
	}

	int32_t num_byte_for_bitmap = (frontier.max_num_elems + 8 * sizeof(uint32_t) - 1)/(sizeof(uint32_t) * 8);

	if (threadIdx.x + blockIdx.x * blockDim.x == 0) 
		frontier.d_num_elems_output[0] = 0;
	parallel_memset((unsigned char*)frontier.d_bit_map_output, 0, sizeof(uint32_t) * num_byte_for_bitmap);		
	this_grid().sync();
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

static void __device__ dedup_frontier_device_perfect(VertexFrontier &frontier) {
	for(int32_t vidx = threadIdx.x + blockDim.x * blockIdx.x; vidx < frontier.d_num_elems_input[0]; vidx += blockDim.x * gridDim.x) {
		int32_t vid = frontier.d_sparse_queue_input[vidx];
		if (writeMax(&frontier.d_dedup_counters[vid], frontier.curr_dedup_counter)) {
			enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, vid);
		}
	}
}
static void __global__ dedup_frontier_kernel_perfect(VertexFrontier frontier) {
	dedup_frontier_device_perfect(frontier);	
}
static void dedup_frontier_perfect(VertexFrontier &frontier) {
	frontier.curr_dedup_counter++;
	dedup_frontier_kernel_perfect<<<NUM_CTA, CTA_SIZE>>>(frontier);
	swap_queues(frontier);
}
bool __device__ true_function(int32_t _) {
	return true;
}
template <bool to_func(int32_t)>
static void __device__ vertex_set_create_reverse_sparse_queue(VertexFrontier &frontier) {
	for (int32_t node_id = blockDim.x * blockIdx.x + threadIdx.x; node_id < frontier.max_num_elems; node_id += blockDim.x * gridDim.x) {
		if ((to_func(node_id)))
			enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, node_id);
	}	
}
template <bool to_func(int32_t)>
static void __global__ vertex_set_create_reverse_sparse_queue_kernel(VertexFrontier frontier) {
	vertex_set_create_reverse_sparse_queue<to_func>(frontier);
}

template <bool to_func(int32_t)>
static void vertex_set_create_reverse_sparse_queue_host(VertexFrontier &frontier) {
	vertex_set_create_reverse_sparse_queue_kernel<to_func><<<NUM_CTA, CTA_SIZE>>>(frontier);
	swap_queues(frontier);	
}

template <bool to_func(int32_t)>
static void __device__ vertex_set_create_reverse_sparse_queue_device(VertexFrontier &frontier) {
	vertex_set_create_reverse_sparse_queue<to_func>(frontier);
	this_grid().sync();
	swap_queues_device(frontier);	
}
static void foo_bar(void) {
}

template <bool where_func(int32_t)>
static void __global__ vertex_set_where_kernel(int32_t num_vertices, VertexFrontier frontier) {

	for (int32_t node_id = blockDim.x * blockIdx.x + threadIdx.x; node_id < num_vertices; node_id += blockDim.x * gridDim.x) {
		if (where_func(node_id)) {
			enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, node_id);
		}
	}

}

template <bool where_func(int32_t)>
static void __host__ vertex_set_where(int32_t num_vertices, VertexFrontier &frontier) {
	vertex_set_where_kernel<where_func><<<NUM_CTA, CTA_SIZE>>>(num_vertices, frontier);
	swap_queues(frontier);
}

}

#endif

