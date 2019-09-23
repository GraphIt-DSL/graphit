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

template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier)>
void __device__ vertex_based_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
	int32_t vid = threadIdx.x + blockDim.x * blockIdx.x;
	if (vid >= input_frontier.d_num_elems_input[0])
		return;
	int32_t src = input_frontier.d_sparse_queue_input[vid];
	for (int32_t eid = graph.d_src_offsets[src]; eid < graph.d_src_offsets[src+1]; eid++) {
		int32_t dst = graph.d_edge_dst[eid];
		load_balance_payload(graph, src, dst, eid, output_frontier);
	}
}

void __host__ vertex_based_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
	int32_t num_threads = builtin_getVertexSetSize(frontier);
	num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
	cta_size = CTA_SIZE;
}


}

#endif
