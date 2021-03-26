#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
int *dist;
swarm::BucketQueue<int> swarm_pq;
int swarm_pq_delta;
void dist_generated_vector_op_apply_func_0(int v) {
	dist[v] = 2147483647;
}
void updateEdge(int src, int dst, int weight, swarm_runtime::VertexFrontier __output_frontier) {
	int new_dist = (dist[src] + weight);
	if (dist[dst] > new_dist) {
		dist[dst] = new_dist;
	};
}
void printDist(int v) {
	swarm_runtime::print(dist[v]);
}
void reset(int v) {
	dist[v] = 2147483647;
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		reset(_iter);
	};
	int start_vertex = atoi(__argv[2]);
	dist[start_vertex] = 0;
	swarm_pq_delta = 1;
	swarm_pq.push_init(0, start_vertex);
	swarm_pq.for_each_prio([](unsigned level, int src, auto push) {
		int32_t edgeZero = edges.h_src_offsets[src];
		int32_t edgeLast = edges.h_src_offsets[src+1];
		for (int i = edgeZero; i < edgeLast; i++) {
			int dst = edges.h_edge_dst[i];
			int weight = edges.h_edge_weight[i];
			{
				int new_dist = (dist[src] + weight);
				if (dist[dst] > new_dist) {
					dist[dst] = new_dist;
					push((new_dist)/swarm_pq_delta, dst);
				};
			}
		}
	}, [](unsigned level, int src) {
	});
}

#include <fstream>
#include <iostream>
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	dist = new int[swarm_runtime::builtin_getVertices(edges)];
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		dist_generated_vector_op_apply_func_0(_iter);
	};
	SCC_PARALLEL( swarm_main(); );

	std::ofstream f("ds_answers.txt");
        if (!f.is_open()) {
                printf("file open failed.\n");
                return -1;
        }

        for (int i = 0; i < swarm_runtime::builtin_getVertices(edges); i++) {
                f << dist[i] << std::endl;
        }
        f.close();	
}
