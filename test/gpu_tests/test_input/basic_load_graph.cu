#include "gpu_intrinsics.h"

gpu_runtime::GraphT<int32_t> edges;

int __host__ main(int argc, char* argv[]) {
	gpu_runtime::load_graph(edges, argv[1], false);
	std::cout << edges.num_vertices << ", " << edges.num_edges << std::endl;	
	return 0;
}
