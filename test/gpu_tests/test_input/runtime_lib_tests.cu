#include <gtest.h>
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
	gpu_runtime::GPUPriorityQueue<int> pq = gpu_runtime::GPUPriorityQueue<int>(priorities);
	EXPECT_EQ (14, num_vertices);
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
