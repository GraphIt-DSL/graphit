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
void updateEdge(int src, int dst, swarm_runtime::VertexFrontier __output_frontier) {
	int src_id = IDs[src];
	int dst_id = IDs[dst];
	bool result_var1 = (bool)0;
	result_var1 = swarm_runtime::min_reduce(IDs[dst_id], IDs[src_id]);
	if (result_var1) {
	}
	bool result_var2 = (bool)0;
	result_var2 = swarm_runtime::min_reduce(IDs[src_id], IDs[dst_id]);
	if (result_var2) {
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
		update[0] = 1;
	}
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
	int n = swarm_runtime::builtin_getVertices(edges);
	swarm::BucketQueue<int> swarm_frontier;
	swarm_runtime::VertexFrontier frontier = swarm_runtime::create_new_vertex_set(swarm_runtime::builtin_getVertices(edges), n);
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		init(_iter);
	};
	for (int i = 0; i < frontier.size(); i++){
		swarm_frontier.push_init(0, frontier[i]);
	}
	swarm_frontier.for_each_prio([](unsigned level, int src, auto push) {
		switch (level % 2) {
		case 0: {
			int32_t edgeZero = edges.h_src_offsets[src];
			int32_t edgeLast = edges.h_src_offsets[src+1];
			for (int i = edgeZero; i < edgeLast; i++) {
				int dst = edges.h_edge_dst[i];
				{
					int src_id = IDs[src];
					int dst_id = IDs[dst];
					bool result_var1 = (bool)0;
					result_var1 = swarm_runtime::min_reduce(IDs[dst_id], IDs[src_id]);
					if (result_var1) {
						push(level + 1, dst_id);
					}
					bool result_var2 = (bool)0;
					result_var2 = swarm_runtime::min_reduce(IDs[src_id], IDs[dst_id]);
					if (result_var2) {
						push(level + 1, src_id);
					}
				}
			}
			break;
		}
		case 1: {
			push(level + 1, src);
			break;
		}
		}
	}, [](unsigned level, int src, auto push) {  // manually inserted the push
		switch (level % 2) {
		case 0: {
			update[0] = 1;
			push(level+1, src);  // manually inserted
			break;
		}
		case 1: {
			while ((update[0]) != (0)) {
				update[0] = 0;
				for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
					pjump(_iter);
				};
			}
			break;
		}
	}
});
swarm_runtime::clear_frontier(frontier);
	swarm_runtime::deleteObject(frontier);
}
#include <iostream>
#include <fstream>
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	IDs = new int[swarm_runtime::builtin_getVertices(edges)];
	update = new int[1];
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
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
