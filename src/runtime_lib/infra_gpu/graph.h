#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

// GraphT data structure 

namespace gpu_runtime {

template <typename EdgeWeightType>
struct GraphT { // Field names are according to CSR, reuse for CSC
	int32_t num_vertices;
	int32_t num_edges;

	// Host pointers
	int32_t *h_src_offsets; // num_vertices + 1;
	int32_t *h_edge_src; // num_edges;
	int32_t *h_edge_dst; // num_edges;
	EdgeWeightType *h_edge_weight; // num_edges;

	// Device pointers
	int32_t *d_src_offsets; // num_vertices + 1;
	int32_t *d_edge_src; // num_edges;
	int32_t *d_edge_dst; // num_edges;
	EdgeWeightType *d_edge_weight; // num_edges;

	int32_t h_get_degree(int32_t vertex_id) {
		return h_src_offsets[vertex_id + 1] - h_src_offsets[vertex_id];
	}
	int32_t d_get_degree(int32_t vertex_id) {
		return d_src_offsets[vertex_id + 1] - d_src_offsets[vertex_id];
	}
};


template <typename EdgeWeightType>
void load_graph(GraphT<EdgeWeightType> &graph, std::string filename, bool to_sort = false);


}
#endif
