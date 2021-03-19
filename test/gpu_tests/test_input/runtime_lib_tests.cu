#include <gtest.h>
#define NUM_BLOCKS (80)
#include "gpu_intrinsics.h"

std::string graph_directory;

class GPURuntimeLibTest: public ::testing::Test {
protected:
	virtual void SetUp() {
	}
	virtual void TearDown() {
	}
	
};
TEST_F(GPURuntimeLibTest, SimpleLoadGraphFromFileTest) {
	gpu_runtime::GraphT<int32_t> edges;
	gpu_runtime::load_graph(edges, graph_directory + "/simple_mtx.mtx", false);
	EXPECT_EQ (14, edges.num_vertices);
}

TEST_F(GPURuntimeLibTest, SimplePriorityQueueTest){
	gpu_runtime::GraphT<int32_t> edges;
	gpu_runtime::load_graph(edges, graph_directory + "/simple_mtx.mtx", false);
	int num_vertices = gpu_runtime::builtin_getVertices(edges);
	int* priorities = new int[num_vertices]; 
	gpu_runtime::GPUPriorityQueue<int> pq;
	EXPECT_EQ (14, num_vertices);
}

__device__ int32_t* test_array_1;
void __device__ vertex_set_apply_all_test_function(int32_t vid) {
	test_array_1[vid] += 1;
}

TEST_F(GPURuntimeLibTest, VertexSetApplyAllTest) {
	gpu_runtime::GraphT<int32_t> edges;
	gpu_runtime::load_graph(edges, graph_directory + "/simple_mtx.mtx", false);
	int num_vertices = gpu_runtime::builtin_getVertices(edges);
	EXPECT_EQ (14, num_vertices);
	
	int32_t *test_array;	
	cudaMalloc(&test_array, num_vertices * sizeof(int32_t));
	cudaMemcpyToSymbol(test_array_1, &test_array, sizeof(int32_t*), 0);
	
	int32_t *test_array_host = new int32_t[num_vertices];
	cudaMemset(test_array, 0, sizeof(int32_t) * num_vertices);	

	gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorAll, vertex_set_apply_all_test_function><<<NUM_CTA, CTA_SIZE>>>(edges.getFullFrontier());
	
	cudaMemcpy(test_array_host, test_array, sizeof(int32_t) * num_vertices, cudaMemcpyDeviceToHost);
	cudaFree(test_array);
	for (int32_t index = 0; index < num_vertices; index++) {
		EXPECT_EQ(1, test_array_host[index]);
	}	
}


TEST_F(GPURuntimeLibTest, VertexSetApplySparseTest) {
	gpu_runtime::GraphT<int32_t> edges;
	gpu_runtime::load_graph(edges, graph_directory + "/simple_mtx.mtx", false);
	int num_vertices = gpu_runtime::builtin_getVertices(edges);
	EXPECT_EQ (14, num_vertices);
	
	int32_t *test_array;	
	cudaMalloc(&test_array, num_vertices * sizeof(int32_t));
	cudaMemcpyToSymbol(test_array_1, &test_array, sizeof(int32_t*), 0);
	
	int32_t *test_array_host = new int32_t[num_vertices];
	cudaMemset(test_array, 0, sizeof(int32_t) * num_vertices);	

	gpu_runtime::VertexFrontier frontier = gpu_runtime::create_new_vertex_set(num_vertices);

	builtin_addVertex(frontier, 0);
	builtin_addVertex(frontier, 7);
	builtin_addVertex(frontier, 13);

	
	gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorSparse, vertex_set_apply_all_test_function><<<NUM_CTA, CTA_SIZE>>>(frontier);
	
	cudaMemcpy(test_array_host, test_array, sizeof(int32_t) * num_vertices, cudaMemcpyDeviceToHost);
	cudaFree(test_array);
	for (int32_t index = 0; index < num_vertices; index++) {
		if (index == 0 || index == 7 || index == 13) 
			EXPECT_EQ(1, test_array_host[index]);
		else 
			EXPECT_EQ(0, test_array_host[index]);
	}	
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cout << "Test needs path to graph directory as first argument" << std::endl;
		exit(-1);
	}
	graph_directory = argv[1];
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();	
}
