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
		swarm_runtime::builtin_addVertex(__output_frontier, dst_id);
	}
	bool result_var2 = (bool)0;
	result_var2 = swarm_runtime::min_reduce(IDs[src_id], IDs[dst_id]);
	if (result_var2) {
		swarm_runtime::builtin_addVertex(__output_frontier, src_id);
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
	swarm::UnorderedQueue<int>* frontier = new swarm::UnorderedQueue<int>();
	frontier->init(swarm_runtime::builtin_getVertices(edges));
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		init(_iter);
	};
	for (int i = 0, m = n; i < m; i++) {
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
				{
					swarm::UnorderedQueue<int>* __output_frontier = output;
					int src_id = IDs[src];
					int dst_id = IDs[dst];
					bool result_var1 = (bool)0;
					result_var1 = swarm_runtime::min_reduce(IDs[dst_id], IDs[src_id]);
					if (result_var1) {
						swarm_runtime::builtin_addVertex(__output_frontier, dst_id);
					}
					bool result_var2 = (bool)0;
					result_var2 = swarm_runtime::min_reduce(IDs[src_id], IDs[dst_id]);
					if (result_var2) {
						swarm_runtime::builtin_addVertex(__output_frontier, src_id);
					}
				}
			}
		});
		swarm_runtime::deleteObject(frontier);
		frontier = output;
		update[0] = 1;
		while ((update[0]) != (0)) {
			update[0] = 0;
			for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
				pjump(_iter);
			};
		}
	}
	swarm_runtime::deleteObject(frontier);
}
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	IDs = new int[swarm_runtime::builtin_getVertices(edges)];
	update = new int[1];
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		IDs_generated_vector_op_apply_func_0(_iter);
	};
	SCC_PARALLEL( swarm_main(); );
}
