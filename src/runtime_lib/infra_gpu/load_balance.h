#ifndef GRAPHIT_GPU_LOAD_BALANCE_H
#define GRAPHIT_GPU_LOAD_BALANCE_H

namespace gpu_runtime {
template <typename EdgeWeightType, void load_balance_payload (GraphT<EdgeWeightType>, int32_t, int32_t, int32_t, VertexFrontier)>
void __device__ vertex_based_load_balance(GraphT<EdgeWeightType> graph, VertexFrontier input_frontier, VertexFrontier output_frontier) {
}

void __host__ vertex_based_load_balance_info(VertexFrontier &frontier, int32_t &num_cta, int32_t &cta_size) {
}


}

#endif
