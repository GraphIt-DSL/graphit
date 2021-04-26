#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
int *parent;

bool* inFrontier;
void parent_generated_vector_op_apply_func_0(int v) {
	parent[v] = -(1);
}
void updateEdge(int src, int dst, swarm_runtime::VertexFrontier __output_frontier) {
	parent[dst] = src;
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
	swarm::PrioQueue<int> swarm_frontier;
	swarm_runtime::VertexFrontier frontier = swarm_runtime::create_new_vertex_set(swarm_runtime::builtin_getVertices(edges), 0);
	int start_vertex = atoi(__argv[2]);
	swarm_runtime::builtin_addVertex(frontier, start_vertex);
	parent[start_vertex] = start_vertex;
	for (int i = 0, m = frontier.size(); i < m; i++){
		swarm_frontier.push_init(0, frontier[i]);
	}
	swarm_frontier.for_each_prio([](unsigned level, int src, auto push) {
		int32_t edgeZero = edges.h_src_offsets[src];
		int32_t edgeLast = edges.h_src_offsets[src+1];
		SCC_OPT_LOOP_COARSEN_FACTOR(SWARM_CACHE_LINE/sizeof(int))  // MANUALLY ADDED
		for (int i = edgeZero; i < edgeLast; i++) {
			int dst = edges.h_edge_dst[i];
			auto* parent_addr = &parent[dst];  // MANUALLY ADDED
			SCC_OPT_TASK();  // MANUALLY ADDED
			SCC_OPT_CACHELINEHINT(parent_addr);  // MANUALLY ADDED
			if (toFilter(dst)) {
				{
					parent[dst] = src;
					push(level + 1, dst);
				}
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
