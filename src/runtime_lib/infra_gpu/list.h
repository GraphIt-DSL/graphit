#ifndef GRAPHIT_GPU_LIST_H
#define GRAPHIT_GPU_LIST_H

#include <vector>
#include <cooperative_groups.h>
using namespace cooperative_groups;

namespace gpu_runtime {
/*
template <typename T>
static void builtin_append(std::vector<T> &vec, T elem) {
	vec.push_back(elem);	
}

template <typename T>
static T builtin_pop(std::vector<T> &vec) {
	T ret = vec.back();
	vec.pop_back();
	return ret;
}
*/

class VertexFrontierList {
public:
	int32_t max_num_elems; 
	int32_t current_levels;
	
	int32_t * d_level_indices;
	int32_t * d_vertices;	
};

VertexFrontierList create_new_vertex_frontier_list(int32_t max_elems) {
	VertexFrontierList vl;
	vl.max_num_elems = max_elems;
	vl.current_levels = 0;
	
	cudaMalloc(&(vl.d_level_indices), sizeof(int32_t) * (max_elems + 1));	
	//vl.h_level_indices = new int32_t [max_elems + 1];	
	//vl.h_level_indices[0] = 0;
	cudaMemset(vl.d_level_indices, 0, sizeof(int32_t));
	cudaMalloc(&(vl.d_vertices), sizeof(int32_t) * max_elems);
	return vl;
}


void builtin_insert(VertexFrontierList &vl, VertexFrontier &frontier) {
	int32_t array[2];

	cudaMemcpy(array, vl.d_level_indices + vl.current_levels, sizeof(int32_t), cudaMemcpyDeviceToHost);
	vertex_set_prepare_sparse(frontier);	
	frontier.format_ready = VertexFrontier::SPARSE;
	//int32_t at = vl.h_level_indices[vl.current_levels];
	int32_t at = array[0];
	int32_t num_elems = builtin_getVertexSetSize(frontier);
	cudaMemcpy(vl.d_vertices + at, frontier.d_sparse_queue_input, num_elems * sizeof(int32_t), cudaMemcpyDeviceToDevice);
	//vl.h_level_indices[vl.current_levels + 1] = at + num_elems;	
	array[1] = at + num_elems;

	cudaMemcpy(vl.d_level_indices + vl.current_levels + 1, array + 1, sizeof(int32_t), cudaMemcpyHostToDevice);
	vl.current_levels++;
}

void __device__ device_builtin_insert(VertexFrontierList &vl, VertexFrontier &frontier) {
	vertex_set_prepare_sparse_device(frontier);
	frontier.format_ready = VertexFrontier::SPARSE;

	int32_t at = vl.d_level_indices[vl.current_levels];
	int32_t num_elems = device_builtin_getVertexSetSize(frontier);
	parallel_memcpy((unsigned char*)(vl.d_vertices + at), (unsigned char*)(frontier.d_sparse_queue_input), num_elems * sizeof(int32_t));
	if (threadIdx.x == 0 && blockIdx.x == 0)
		vl.d_level_indices[vl.current_levels + 1] = at + num_elems;
	vl.current_levels++;
	this_grid().sync();
}


void builtin_retrieve(VertexFrontierList &vl, VertexFrontier &frontier) {
	if (vl.current_levels == 0) {
		assert(false && "Too deep into vertex frontier list");
	}	
	int32_t array[2];

	cudaMemcpy(array, vl.d_level_indices + vl.current_levels - 1, sizeof(int32_t)*2, cudaMemcpyDeviceToHost);
	//int32_t at = vl.h_level_indices[vl.current_levels - 1];
	//int32_t num_elems = vl.h_level_indices[vl.current_levels] - at;
	int32_t at = array[0];
	int32_t num_elems = array[1] - at;
	cudaMemcpy(frontier.d_sparse_queue_input, vl.d_vertices + at, num_elems * sizeof(int32_t), cudaMemcpyDeviceToDevice);
	cudaMemcpy(frontier.d_num_elems_input, &num_elems, sizeof(int32_t), cudaMemcpyHostToDevice);
	frontier.format_ready = gpu_runtime::VertexFrontier::SPARSE;
	vl.current_levels--;
}
void __device__ device_builtin_retrieve(VertexFrontierList &vl, VertexFrontier &frontier) {
	if (vl.current_levels == 0)
		assert(false && "Too deep into vertex frontier list");
	int32_t at = vl.d_level_indices[vl.current_levels -1];		
	int32_t num_elems = vl.d_level_indices[vl.current_levels] - at;
	parallel_memcpy((unsigned char*)frontier.d_sparse_queue_input, (unsigned char*) (vl.d_vertices + at), num_elems * sizeof(int32_t));
	if (threadIdx.x == 0 && blockIdx.x == 0)
		frontier.d_num_elems_input[0] = num_elems;
	frontier.format_ready = gpu_runtime::VertexFrontier::SPARSE;
	vl.current_levels--;
	this_grid().sync();
}
}


#endif
