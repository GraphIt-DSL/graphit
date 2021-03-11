#ifndef SWARM_GRAPH_H
#define SWARM_GRAPH_H

// GraphT data structure 
#define IGNORE_JULIENNE_TYPES
#include <assert.h>
#include <iostream>
#include <cstring>

typedef int int32_t;

namespace swarm_runtime {

template <typename EdgeWeightType>
struct GraphT { // Field names are according to CSR, reuse for CSC
        typedef EdgeWeightType EdgeWeightT;
        int32_t num_vertices;
        int32_t num_edges;

        // Host pointers
        int32_t *h_src_offsets; // num_vertices + 1;
        int32_t *h_edge_src; // num_edges;
        int32_t *h_edge_dst; // num_edges;
        EdgeWeightType *h_edge_weight; // num_edges;
	int32_t *out_degrees;
	GraphT<EdgeWeightType> *transposed_graph;

        int32_t h_get_degree(int32_t vertex_id) {
                return h_src_offsets[vertex_id + 1] - h_src_offsets[vertex_id];
        }
};

template <typename EdgeWeightType>
int builtin_getVertices(GraphT<EdgeWeightType> &graph) {
  return graph.num_vertices;
}

template <typename EdgeWeightType>
int* builtin_getOutDegrees(GraphT<EdgeWeightType> &graph) {
  if (graph.out_degrees == nullptr) {
    graph.out_degrees = new int[graph.num_vertices];
    for (int i = 0; i < graph.num_vertices; i++) {
      graph.out_degrees[i] = graph.h_src_offsets[i+1] - graph.h_src_offsets[i];
    }
  }
  return graph.out_degrees;
}

void consume(int32_t _) {
}
#define CONSUME consume
template <typename EdgeWeightType>
void static sort_with_degree(GraphT<EdgeWeightType> &graph) {
        assert(false && "Sort with degree not yet implemented\n");
        return;
}
static bool string_ends_with(const char* str, const char* sub_str) {
        if (strlen(sub_str) > strlen(str))
                return false;
        int32_t len1 = strlen(str);
        int32_t len2 = strlen(sub_str);
        if (strcmp(str + len1 - len2, sub_str) == 0)
                return true;
        return false;
}

template <typename EdgeWeightType>
static void load_graph(GraphT<EdgeWeightType> &graph, std::string filename, bool to_sort = false) {
        const char* bin_extension = to_sort?".graphit_sbin":".graphit_bin";
        char bin_filename[1024];
        strcpy(bin_filename, filename.c_str());

        if (string_ends_with(filename.c_str(), bin_extension) == false)  {
                strcat(bin_filename, ".");
                strcat(bin_filename, typeid(EdgeWeightType).name());
                strcat(bin_filename, bin_extension);
        }

        FILE *bin_file = fopen(bin_filename, "rb");
        if (!bin_file && string_ends_with(filename.c_str(), bin_extension)) {
                std::cout << "Binary file not found" << std::endl;
                exit(-1);
        }
        if (bin_file) {
                CONSUME(fread(&graph.num_vertices, sizeof(int32_t), 1, bin_file));
                CONSUME(fread(&graph.num_edges, sizeof(int32_t), 1, bin_file));

                graph.h_edge_src = new int32_t[graph.num_edges];
                graph.h_edge_dst = new int32_t[graph.num_edges];
                graph.h_edge_weight = new EdgeWeightType[graph.num_edges];

                graph.h_src_offsets = new int32_t[graph.num_vertices + 1];

                CONSUME(fread(graph.h_edge_src, sizeof(int32_t), graph.num_edges, bin_file));
                CONSUME(fread(graph.h_edge_dst, sizeof(int32_t), graph.num_edges, bin_file));
                CONSUME(fread(graph.h_edge_weight, sizeof(EdgeWeightType), graph.num_edges, bin_file));

                CONSUME(fread(graph.h_src_offsets, sizeof(int32_t), graph.num_vertices + 1, bin_file));
                fclose(bin_file);
        } else {
                assert(false && "Only bin files are supported\n");
	}
}

template <typename EdgeWeightType>
static GraphT<EdgeWeightType> builtin_transpose(GraphT<EdgeWeightType> &graph) {
	if (graph.transposed_graph != nullptr)
		return *(graph.transposed_graph);
	// For now we will return the same graph
	GraphT<EdgeWeightType> output_graph;
	output_graph.num_vertices = graph.num_vertices;
	output_graph.num_edges = graph.num_edges;
	
	output_graph.h_src_offsets = new int32_t[graph.num_vertices+2];
	output_graph.h_edge_src = new int32_t[graph.num_edges];
	output_graph.h_edge_dst = new int32_t[graph.num_edges];
	output_graph.h_edge_weight = new EdgeWeightType[graph.num_edges];
	
	for (int32_t i = 0; i < graph.num_vertices + 2; i++)
		output_graph.h_src_offsets[i] = 0;
	
	// This will count the degree for each vertex in the transposed graph
	for (int32_t i = 0; i < graph.num_edges; i++) {
		int32_t dst = graph.h_edge_dst[i];
		output_graph.h_src_offsets[dst+2]++;
	}

	// We will now create cummulative sums
	for (int32_t i = 0; i < graph.num_vertices; i++) {
		output_graph.h_src_offsets[i+2] += output_graph.h_src_offsets[i+1];	
	}
	
	// Finally fill in the edges and the weights for the new graph		
	for (int32_t i = 0; i < graph.num_edges; i++) {
		int32_t dst = graph.h_edge_dst[i];
		int32_t pos = output_graph.h_src_offsets[dst+1];
		output_graph.h_src_offsets[dst+1]++;
		output_graph.h_edge_src[pos] = dst;
		output_graph.h_edge_dst[pos] = graph.h_edge_src[i];
		output_graph.h_edge_weight[pos] = graph.h_edge_weight[i];
	}
	output_graph.transposed_graph = &graph;
	graph.transposed_graph = new GraphT<EdgeWeightType>(output_graph);

	
	return output_graph;
}
}
#endif
