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
	swarm_runtime::min_reduce(IDs[dst_id], IDs[src_id]);
	swarm_runtime::min_reduce(IDs[src_id], IDs[dst_id]);
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
	for(int trail = 0; trail < 10; trail++) {
		swarm::BucketQueue<int> swarm_frontier;
		swarm_runtime::VertexFrontier frontier = swarm_runtime::create_new_vertex_set(swarm_runtime::builtin_getVertices(edges), n);
		swarm_runtime::startTimer();
		for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
			init(_iter);
		};
		for (int i = 0; i < frontier.size(); i++){
			swarm_frontier.push_init(0, frontier[i]);
		}
		swarm_frontier.for_each_prio([](unsigned level, int src, auto push) {
			int32_t edgeZero = edges.h_src_offsets[src];
			int32_t edgeLast = edges.h_src_offsets[src+1];
			for (int i = edgeZero; i < edgeLast; i++) {
				int dst = edges.h_edge_dst[i];
				{
					int src_id = IDs[src];
					int dst_id = IDs[dst];
					if ( ( IDs[dst_id]) > ( IDs[src_id]) ) { 
						IDs[dst_id] = IDs[src_id];
						push(level + 1, IDs[dst_id]);
					}
					if ( ( IDs[src_id]) > ( IDs[dst_id]) ) { 
						IDs[src_id] = IDs[dst_id];
						push(level + 1, IDs[src_id]);
					}
				}
			}
			update[0] = 1;
			while ((update[0]) != (0)) {
				update[0] = 0;
				pjump(src);
;
			}
		}, [](unsigned level, int src) {
		});
		swarm_runtime::clear_frontier(frontier);
		float elapsed_time = swarm_runtime::stopTimer();
		deleteObject(frontier);
		swarm_runtime::print("elapsed time: ");
		swarm_runtime::print(elapsed_time);
	}
}
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
}
