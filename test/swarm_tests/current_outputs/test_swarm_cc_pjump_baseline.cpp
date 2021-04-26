#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
int *IDs;
int *update;
void IDs_generated_vector_op_apply_func_0(int v) {
	IDs[v] = 1;
}
void updateEdge(int src, int dst) {
	int src_id = IDs[src];
	int dst_id = IDs[dst];
	int p_src_id = IDs[src_id];
	int p_dst_id = IDs[dst_id];
	swarm_runtime::min_reduce(IDs[dst_id], p_src_id);
	swarm_runtime::min_reduce(IDs[src_id], p_dst_id);
	if ((update[16]) == (0)) {
		if ((p_dst_id) != (IDs[dst_id])) {
			update[16] = 1;
		}
		if ((p_src_id) != (IDs[src_id])) {
			update[16] = 1;
		}
	}
}
void init(int v) {
	IDs[v] = v;
}
void pjump(int v) {
	int y = IDs[v];
	int x = IDs[y];
	if ((x) != (y)) {
		IDs[v] = x;
		if (update[0] == 0) {
			update[0] = 1;
		}
	}
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
        int n = swarm_runtime::builtin_getVertices(edges);
        for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
                init(_iter);
        };
        update[16] = 1;
        while ((update[16]) != (0)) {
                update[16] = 0;
                for (int _iter = 0, m = edges.num_edges; _iter < m; _iter++) {
                        int _src = edges.h_edge_src[_iter];
                        int _dst = edges.h_edge_dst[_iter];
                        updateEdge(_src, _dst);
                };
                update[0] = 1;
                while ((update[0]) != (0)) {
                        update[0] = 0;
                        for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
                                pjump(_iter);
                        };
                }
        }
}

#include <iostream>
#include <fstream>
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	IDs = new int[swarm_runtime::builtin_getVertices(edges)];
	update = new int[17];
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		IDs_generated_vector_op_apply_func_0(_iter);
	};
	SCC_PARALLEL( swarm_main(); );
	std::ofstream f("cc_answers.txt");
        if (!f.is_open()) {
                printf("file open failed.\n");
                return -1;
        }
        for (int i = 0; i < swarm_runtime::builtin_getVertices(edges); i++) {
                f << IDs[i] << std::endl;
        }
        f.close();	
}
