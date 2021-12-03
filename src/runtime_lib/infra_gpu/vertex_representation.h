#ifndef VERTEX_REPRESENTATION_H
#define VERTEX_REPRESENTATION_H

#include "infra_gpu/vertex_frontier.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;
namespace gpu_runtime {
template <typename AccessorType, bool condition(VertexFrontier&, int32_t), void update(VertexFrontier&, int32_t)>
static void __device__ generalized_prepare_from_to(VertexFrontier &frontier) {
	int32_t total_work = AccessorType::getSize(frontier);
	for (int32_t index = threadIdx.x + blockIdx.x * blockDim.x; index < total_work; index += gridDim.x * blockDim.x) {
		int32_t node_id = AccessorType::getElement(frontier, index);
		if (condition(frontier, node_id))
			update(frontier, node_id);	
	}
}

template <typename AccessorType, bool condition(VertexFrontier&, int32_t), void update(VertexFrontier&, int32_t)>
static void __global__ generalized_prepare_from_to_kernel(VertexFrontier frontier) {
	generalized_prepare_from_to<AccessorType, condition, update>(frontier);
}

static bool __device__ condition_sparse(VertexFrontier &frontier, int32_t node_id) {
	return true;
}
static bool __device__ condition_bytemap(VertexFrontier &frontier, int32_t node_id) {
	return frontier.d_byte_map_input[node_id] == 1;	
}
static bool __device__ condition_bitmap(VertexFrontier &frontier, int32_t node_id) {
	return checkBit(frontier.d_bit_map_input, node_id);
}


static void __device__ update_sparse(VertexFrontier &frontier, int32_t node_id) {
	enqueueVertexSparseQueue(frontier.d_sparse_queue_output, frontier.d_num_elems_output, node_id);
} 

static void __device__ update_bytemap(VertexFrontier &frontier, int32_t node_id) {
	enqueueVertexBytemap(frontier.d_byte_map_output, frontier.d_num_elems_output, node_id);
}

static void __device__ update_bitmap(VertexFrontier &frontier, int32_t node_id) {
	enqueueVertexBitmap(frontier.d_bit_map_output, frontier.d_num_elems_output, node_id);
}

static void vertex_set_prepare_sparse(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		generalized_prepare_from_to_kernel<AccessorAll, condition_bytemap, update_sparse><<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_queues(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		generalized_prepare_from_to_kernel<AccessorAll, condition_bitmap, update_sparse><<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_queues(frontier);
		return;
	}
}
static void __device__ vertex_set_prepare_sparse_device(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		generalized_prepare_from_to<AccessorAll, condition_bytemap, update_sparse>(frontier);
		this_grid().sync();
		swap_queues_device(frontier);
		this_grid().sync();
		return;
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		generalized_prepare_from_to<AccessorAll, condition_bitmap, update_sparse>(frontier);
		this_grid().sync();
		swap_queues_device(frontier);
		this_grid().sync();
		return;
	}
}
static void vertex_set_prepare_boolmap(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		generalized_prepare_from_to_kernel<AccessorSparse, condition_sparse, update_bytemap><<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bytemaps(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		return;
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		generalized_prepare_from_to_kernel<AccessorAll, condition_bitmap, update_bytemap><<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bytemaps(frontier);
		return;
	}
}
static void __device__ vertex_set_prepare_boolmap_device(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		generalized_prepare_from_to<AccessorSparse, condition_sparse, update_bytemap>(frontier);
		this_grid().sync();
		swap_bytemaps_device(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		return;
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		generalized_prepare_from_to<AccessorAll, condition_bitmap, update_bytemap>(frontier);
		this_grid().sync();
		swap_bytemaps_device(frontier);
		return;
	}
}
static void vertex_set_prepare_bitmap(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		generalized_prepare_from_to_kernel<AccessorSparse, condition_sparse, update_bitmap><<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bitmaps(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		generalized_prepare_from_to_kernel<AccessorAll, condition_bytemap, update_bitmap><<<NUM_CTA, CTA_SIZE>>>(frontier);
		swap_bitmaps(frontier);
		return;	
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		return;
	}
}
static void __device__ vertex_set_prepare_bitmap_device(VertexFrontier &frontier) {
	if (frontier.format_ready == VertexFrontier::SPARSE) {
		generalized_prepare_from_to<AccessorSparse, condition_sparse, update_bitmap>(frontier);
		this_grid().sync();
		swap_bitmaps_device(frontier);
		return;
	} else if (frontier.format_ready == VertexFrontier::BYTEMAP) {
		generalized_prepare_from_to<AccessorAll, condition_bytemap, update_bitmap>(frontier);
		this_grid().sync();
		swap_bitmaps_device(frontier);
		return;	
	} else if (frontier.format_ready == VertexFrontier::BITMAP) {
		return;
	}

}
}
#endif
