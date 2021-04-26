#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
int *parent;
void parent_generated_vector_op_apply_func_0(int v) {
	parent[v] = -(1);
}
void updateEdge(int src, int dst, swarm_runtime::VertexFrontier __output_frontier) {
	parent[dst] = src;
	swarm_runtime::builtin_addVertex(__output_frontier, dst);
}
bool toFilter(int v) {
	bool output;
	output = (parent[v]) == (-(1));
	return output;
}
void reset(int v) {
	parent[v] = -(1);
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		reset(_iter);
	};
	swarm::UnorderedQueue<int>* frontier = new swarm::UnorderedQueue<int>();
	frontier->init(swarm_runtime::builtin_getVertices(edges));
	int start_vertex = atoi(__argv[2]);
	swarm_runtime::builtin_addVertex(frontier, start_vertex);
	parent[start_vertex] = start_vertex;
	for (int i = 0, m = 0; i < m; i++) {
		frontier->push(i);
	}
	while ((swarm_runtime::builtin_getVertexSetSize(frontier)) != (0)) {
		swarm::UnorderedQueue<int>* output = new swarm::UnorderedQueue<int>();
		output->init(swarm_runtime::builtin_getVertices(edges));
		frontier->for_each([output](int src) {
			int32_t edgeZero = edges.h_src_offsets[src];
			int32_t edgeLast = edges.h_src_offsets[src+1];
			for (int j = edgeZero; j < edgeLast; j++) {
				int dst = edges.h_edge_dst[j];
				if (toFilter(dst)) {
					{
						swarm::UnorderedQueue<int>* __output_frontier = output;
						parent[dst] = src;
						swarm_runtime::builtin_addVertex(__output_frontier, dst);
					}
				}
			}
		});
		swarm_runtime::deleteObject(frontier);
		frontier = output;
	}
	swarm_runtime::deleteObject(frontier);
}

#include <iostream>
#include <fstream>
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	parent = new int[swarm_runtime::builtin_getVertices(edges)];
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		parent_generated_vector_op_apply_func_0(_iter);
	};
	SCC_PARALLEL( swarm_main(); );
	std::ofstream f("bfs_answers.txt");
        if (!f.is_open()) {
                printf("file open failed.\n");
                return -1;
        }
        for (int i = 0; i < swarm_runtime::builtin_getVertices(edges); i++) {
                f << parent[i] << std::endl;
        }
        f.close();
}
