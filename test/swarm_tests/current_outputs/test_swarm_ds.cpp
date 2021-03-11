#include <tuple>
#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
int *dist;
swarm::PrioQueue<int> swarm_pq;
int swarm_pq_delta;
void dist_generated_vector_op_apply_func_0(int v) {
	dist[v] = 2147483647;
}
bool updateEdge(int src, int dst, int weight) {
	bool output3;
	bool dist_trackving_var_2 = (bool)0;
	int new_dist = (dist[src] + weight);
	if (dist[dst] > new_dist) {
		dist[dst] = new_dist;
		push((new_dist)/swarm_pq_delta, dst);
	};
	output3 = dist_trackving_var_2;
	return output3;
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
	for (int i = 0; i < pq.size(); i++){
		swarm_pq.push_init(0, pq[i]);
	}
	swarm_pq.for_each_prio([](unsigned level, int src, auto push) {
		int frontier = getBucketWithGraphItVertexSubset(src);
		int modified_vertexsubset1 = 		int32_t edgeZero = edges.h_src_offsets[src];
		int32_t edgeLast = edges.h_src_offsets[src+1];
		for (int i = edgeZero; i < edgeLast; i++) {
			int dst = edges.h_edge_dst[i];
			{
				bool output3;
				bool dist_trackving_var_2 = (bool)0;
				int new_dist = (dist[src] + weight);
				if (dist[dst] > new_dist) {
					dist[dst] = new_dist;
					push((new_dist)/swarm_pq_delta, dst);
				};
				output3 = dist_trackving_var_2;
			}
		}
;
		deleteObject(frontier);
	}, [](unsigned level) {
	});
	swarm_runtime::clear_frontier(pq);
}
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	dist = new int[swarm_runtime::builtin_getVertices(edges)];
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		dist_generated_vector_op_apply_func_0(_iter);
	};
		SCC_PARALLEL( swarm_main(); );
}
