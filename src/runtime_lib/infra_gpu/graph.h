#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

#include <assert.h>
#include "infra_gpu/support.h"

// GraphT data structure 
#define IGNORE_JULIENNE_TYPES
#include "infra_gapbs/benchmark.h"
#include "infra_gpu/vertex_frontier.h"
#include "graphit_timer.h"
#ifndef FRONTIER_MULTIPLIER
	#define FRONTIER_MULTIPLIER (6)
#endif
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

	GraphT<EdgeWeightType> *transposed_graph;
	

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
	VertexFrontier& __device__ getFullFrontierDevice(void) {
		full_frontier.max_num_elems = num_vertices;
		return full_frontier;	
	}


	// Load balance scratch pads
	// TWC bins
	int32_t *twc_small_bin;
	int32_t *twc_mid_bin;
	int32_t *twc_large_bin;
	
	int32_t *twc_bin_sizes;

	// strict frontiers
	int32_t *strict_sum;
	int32_t *strict_cta_sum;
	int32_t *strict_grid_sum;


	// blocking related parameters
	int32_t num_buckets;
	int32_t *h_bucket_sizes;
	int32_t *d_bucket_sizes;

		
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

static int32_t identify_block_id (int32_t vid, int32_t blocking_size) {
	return vid / blocking_size;
}
template <typename EdgeWeightType>
static void block_graph_edges(GraphT<EdgeWeightType> &input_graph, GraphT<EdgeWeightType> &output_graph, int32_t blocking_size) {
	output_graph = input_graph;
	output_graph.h_src_offsets = nullptr;
	output_graph.d_src_offsets = nullptr;

	output_graph.h_edge_src = new int32_t[input_graph.num_edges];
	output_graph.h_edge_dst = new int32_t[input_graph.num_edges];
	output_graph.h_edge_weight = new EdgeWeightType[input_graph.num_edges];

	int32_t num_blocks = (input_graph.num_vertices + blocking_size - 1)/blocking_size;
	//std::cout << "num blocks " << num_blocks << std::endl;	
	int32_t *block_sizes = new int32_t[num_blocks+1];		
	for (int32_t id = 0; id < num_blocks+1; id++)
		block_sizes[id] = 0;
	
	for (int32_t eid = 0; eid < input_graph.num_edges; eid++) {
		int32_t dst = input_graph.h_edge_dst[eid];
		int32_t block_id = identify_block_id(dst, blocking_size);
		block_sizes[block_id+1] += 1;
	}	
	int32_t running_sum = 0;
	for (int32_t bid = 0; bid < num_blocks; bid++) {
		running_sum += block_sizes[bid];
		block_sizes[bid] = running_sum;
	}
	block_sizes[0] = 0;
	for (int32_t eid = 0; eid < input_graph.num_edges; eid++) {
		int32_t dst = input_graph.h_edge_dst[eid];
		int32_t block_id = identify_block_id(dst, blocking_size);
		int32_t new_eid = block_sizes[block_id];
		block_sizes[block_id]++;
		output_graph.h_edge_src[new_eid] = input_graph.h_edge_src[eid];	
		output_graph.h_edge_dst[new_eid] = input_graph.h_edge_dst[eid];	
		output_graph.h_edge_weight[new_eid] = input_graph.h_edge_weight[eid];	
	}
	
	//delete[] block_sizes;
	output_graph.num_buckets = num_blocks;
	output_graph.h_bucket_sizes = block_sizes;
	


	cudaFree(input_graph.d_edge_src);
	cudaFree(input_graph.d_edge_dst);
	cudaFree(input_graph.d_edge_weight);

	cudaMalloc(&output_graph.d_edge_src, sizeof(int32_t) * output_graph.num_edges);
	cudaMalloc(&output_graph.d_edge_dst, sizeof(int32_t) * output_graph.num_edges);
	cudaMalloc(&output_graph.d_edge_weight, sizeof(EdgeWeightType) * output_graph.num_edges);
	cudaMalloc(&output_graph.d_bucket_sizes, sizeof(int32_t) * num_blocks);
	
	
	cudaMemcpy(output_graph.d_edge_src, output_graph.h_edge_src, sizeof(int32_t) * output_graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(output_graph.d_edge_dst, output_graph.h_edge_dst, sizeof(int32_t) * output_graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(output_graph.d_edge_weight, output_graph.h_edge_weight, sizeof(EdgeWeightType) * output_graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(output_graph.d_bucket_sizes, output_graph.h_bucket_sizes, sizeof(int32_t) * num_blocks, cudaMemcpyHostToDevice);
		
}

template <typename EdgeWeightType>
static GraphT<EdgeWeightType> builtin_transpose(GraphT<EdgeWeightType> &graph) {
	if (graph.transposed_graph != nullptr)
		return *(graph.transposed_graph);
	// For now we will return the same graph
	// TODO: copy transpose implementation from infra_ CPU
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

	cudaMalloc(&output_graph.d_edge_src, sizeof(int32_t) * graph.num_edges);
	cudaMalloc(&output_graph.d_edge_dst, sizeof(int32_t) * graph.num_edges);
	cudaMalloc(&output_graph.d_edge_weight, sizeof(EdgeWeightType) * graph.num_edges);
	cudaMalloc(&output_graph.d_src_offsets, sizeof(int32_t) * (graph.num_vertices + 1));
	
	
	cudaMemcpy(output_graph.d_edge_src, output_graph.h_edge_src, sizeof(int32_t) * graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(output_graph.d_edge_dst, output_graph.h_edge_dst, sizeof(int32_t) * graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(output_graph.d_edge_weight, output_graph.h_edge_weight, sizeof(EdgeWeightType) * graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(output_graph.d_src_offsets, output_graph.h_src_offsets, sizeof(int32_t) * (graph.num_vertices + 1), cudaMemcpyHostToDevice);
	
/*	
	cudaMalloc(&output_graph.twc_small_bin, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&output_graph.twc_mid_bin, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&output_graph.twc_large_bin, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&output_graph.twc_bin_sizes, 3 * sizeof(int32_t));

	cudaMalloc(&output_graph.strict_sum, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&output_graph.strict_cta_sum, NUM_CTA * 2 * sizeof(int32_t));
	cudaMalloc(&output_graph.strict_grid_sum, sizeof(int32_t));
*/
	output_graph.twc_small_bin = graph.twc_small_bin;
	output_graph.twc_mid_bin = graph.twc_mid_bin;
	output_graph.twc_large_bin = graph.twc_large_bin;
	output_graph.strict_sum = graph.strict_sum;
	output_graph.strict_cta_sum = graph.strict_cta_sum;
	output_graph.strict_grid_sum = output_graph.strict_grid_sum;

	output_graph.transposed_graph = &graph;
	graph.transposed_graph = new GraphT<EdgeWeightType>(output_graph);

	
	return output_graph;
}

template <typename EdgeWeightType>
static void load_graph(GraphT<EdgeWeightType> &graph, std::string filename, bool to_sort = false) {
	int flen = strlen(filename.c_str());
	const char* bin_extension = to_sort?".graphit_sbin":".graphit_bin";
	char bin_filename[1024];
	strcpy(bin_filename, filename.c_str());

	if (string_ends_with(filename.c_str(), bin_extension) == false)	 {
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
		CONSUME(fwrite(graph.h_edge_weight, sizeof(EdgeWeightType), graph.num_edges, bin_file));
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

	cudaMalloc(&graph.twc_small_bin, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&graph.twc_mid_bin, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&graph.twc_large_bin, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&graph.twc_bin_sizes, 3 * sizeof(int32_t));

	cudaMalloc(&graph.strict_sum, graph.num_vertices * FRONTIER_MULTIPLIER * sizeof(int32_t));
	cudaMalloc(&graph.strict_cta_sum, NUM_CTA * 2 * sizeof(int32_t));
	cudaMalloc(&graph.strict_grid_sum, sizeof(int32_t));
	
	graph.transposed_graph = nullptr;

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
