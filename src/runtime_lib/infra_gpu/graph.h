#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

#include <assert.h>
#include "infra_gpu/support.h"

// GraphT data structure 
#define IGNORE_JULIENNE_TYPES
#include "infra_gapbs/benchmark.h"
#include "infra_gpu/vertex_frontier.h"
namespace gpu_runtime {

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

	// Device pointers
	int32_t *d_src_offsets; // num_vertices + 1;
	int32_t *d_edge_src; // num_edges;
	int32_t *d_edge_dst; // num_edges;
	EdgeWeightType *d_edge_weight; // num_edges;

	int32_t h_get_degree(int32_t vertex_id) {
		return h_src_offsets[vertex_id + 1] - h_src_offsets[vertex_id];
	}
	int32_t __device__ d_get_degree(int32_t vertex_id) {
		return d_src_offsets[vertex_id + 1] - d_src_offsets[vertex_id];
	}
	VertexFrontier full_frontier;
	VertexFrontier& getFullFrontier(void) {
		full_frontier.max_num_elems = num_vertices;
		return full_frontier;
	}
};
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
	int flen = strlen(filename.c_str());
	const char* bin_extension = to_sort?".graphit_sbin":".graphit_bin";
	char bin_filename[1024];
	strcpy(bin_filename, filename.c_str());

	if (string_ends_with(filename.c_str(), bin_extension) == false)	
		strcat(bin_filename, bin_extension);
	
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
		CONSUME(fread(graph.h_edge_weight, sizeof(int32_t), graph.num_edges, bin_file));

		CONSUME(fread(graph.h_src_offsets, sizeof(int32_t), graph.num_vertices + 1, bin_file));
		fclose(bin_file);	
	} else {
		CLBase cli (filename);
		WeightedBuilder builder (cli);
		WGraph g = builder.MakeGraph();
		graph.num_vertices = g.num_nodes();
		graph.num_edges = g.num_edges();

		graph.h_edge_src = new int32_t[graph.num_edges];
		graph.h_edge_dst = new int32_t[graph.num_edges];
		graph.h_edge_weight = new EdgeWeightType[graph.num_edges];
		
		graph.h_src_offsets = new int32_t[graph.num_vertices + 1];
		
		int32_t tmp = 0;
		graph.h_src_offsets[0] = tmp;
		for (int32_t i = 0; i < g.num_nodes(); i++) {
			for (auto j: g.out_neigh(i)) {
				graph.h_edge_src[tmp] = i;
				graph.h_edge_dst[tmp] = j.v;
				graph.h_edge_weight[tmp] = j.w;	
				tmp++;
			}
			graph.h_src_offsets[i+1] = tmp;
		}	
		if (to_sort)
			sort_with_degree(graph);
		FILE *bin_file = fopen(bin_filename, "wb");
		CONSUME(fwrite(&graph.num_vertices, sizeof(int32_t), 1, bin_file));
		CONSUME(fwrite(&graph.num_edges, sizeof(int32_t), 1, bin_file));
		CONSUME(fwrite(graph.h_edge_src, sizeof(int32_t), graph.num_edges, bin_file));
		CONSUME(fwrite(graph.h_edge_dst, sizeof(int32_t), graph.num_edges, bin_file));
		CONSUME(fwrite(graph.h_edge_weight, sizeof(int32_t), graph.num_edges, bin_file));
		CONSUME(fwrite(graph.h_src_offsets, sizeof(int32_t), graph.num_vertices + 1, bin_file));
		fclose(bin_file);	
	}
	cudaMalloc(&graph.d_edge_src, sizeof(int32_t) * graph.num_edges);
	cudaMalloc(&graph.d_edge_dst, sizeof(int32_t) * graph.num_edges);
	cudaMalloc(&graph.d_edge_weight, sizeof(EdgeWeightType) * graph.num_edges);
	cudaMalloc(&graph.d_src_offsets, sizeof(int32_t) * (graph.num_vertices + 1));
	
	
	cudaMemcpy(graph.d_edge_src, graph.h_edge_src, sizeof(int32_t) * graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(graph.d_edge_dst, graph.h_edge_dst, sizeof(int32_t) * graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(graph.d_edge_weight, graph.h_edge_weight, sizeof(EdgeWeightType) * graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(graph.d_src_offsets, graph.h_src_offsets, sizeof(int32_t) * (graph.num_vertices + 1), cudaMemcpyHostToDevice);
	//std::cout << filename << " (" << graph.num_vertices << ", " << graph.num_edges << ")" << std::endl;

}
template <typename EdgeWeightType>
static int32_t builtin_getVertices(GraphT<EdgeWeightType> &graph) {
	return graph.num_vertices;
}

template <typename EdgeWeightType>
static int32_t __device__ device_builtin_getVertices(GraphT<EdgeWeightType> &graph) {
	return graph.num_vertices;
}

template <typename EdgeWeightType> 
void __global__ init_degrees_kernel(int32_t *degrees, GraphT<EdgeWeightType> graph) {
	for (int32_t vid = threadIdx.x + blockIdx.x * blockDim.x; vid < graph.num_vertices; vid += gridDim.x * blockDim.x) 
		degrees[vid] = graph.d_get_degree(vid);
}

template <typename EdgeWeightType>
static int32_t* builtin_getOutDegrees(GraphT<EdgeWeightType> &graph) {
	int32_t *degrees = nullptr;
	cudaMalloc(&degrees, sizeof(int32_t) * graph.num_vertices);
	init_degrees_kernel<<<NUM_CTA, CTA_SIZE>>>(degrees, graph);
	return degrees;
}

}
#endif
