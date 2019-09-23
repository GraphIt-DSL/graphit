#ifndef GPU_VERTEX_FRONTIER_H
#define GPU_VERTEX_FRONTIER_H

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

	// Extend this to check the current representation
};
}

#endif

