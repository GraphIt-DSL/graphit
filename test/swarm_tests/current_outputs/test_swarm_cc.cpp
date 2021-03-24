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
bool updateEdge(int src, int dst) {
	bool output2;
	bool IDs_trackving_var_1 = (bool)0;
	IDs_trackving_var_1 = swarm_runtime::min_reduce(IDs[dst], IDs[src]);
	output2 = IDs_trackving_var_1;
	return output2;
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
	int start_vertex = atoi(__argv[2]);
	swarm_runtime::builtin_addVertex(frontier, start_vertex);
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
				bool output2;
				bool IDs_trackving_var_1 = (bool)0;
				if ( ( IDs[dst]) > ( IDs[src]) ) { 
					IDs[dst] = IDs[src];
					push(level + 1, dst);
				}
				if ( ( IDs[src]) > ( IDs[dst]) ) { 
					IDs[src] = IDs[dst];
					push(level + 1, src);
				}
				output2 = IDs_trackving_var_1;
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
	deleteObject(frontier);
}

#include <numeric>

// This is Victor's naive manual C++ translation of the code from
// https://github.com/GraphIt-DSL/graphit/blob/gpu_backend/apps/cc_pjump.gt
__attribute__((noswarmify))
void cc_serial() {
        for (int i = 0, e = edges.num_vertices; i < e; i++) {
                IDs[i] = i;
        }
        std::vector<int> frontier(edges.num_vertices);
        std::iota(frontier.begin(), frontier.end(), 0);
        while (frontier.size() != 0) {
                std::vector<int> output;
                for (int src : frontier) {
                        DEBUG("From %u", src);
                        int32_t edgeZero = edges.h_src_offsets[src];
                        int32_t edgeLast = edges.h_src_offsets[src+1];
                        for (int i = edgeZero; i < edgeLast; i++) {
                                int dst = edges.h_edge_dst[i];
                                int src_id = IDs[src];
                                int dst_id = IDs[dst];
                                DEBUG("src %d ID %d, dst %d ID %d",
                                      src, src_id, dst, dst_id);
                                if (IDs[src_id] < IDs[dst_id]) {
                                        DEBUG("Decrease dst's ID from %d to %d",
                                              IDs[dst_id], IDs[src_id]);
                                        IDs[dst_id] = IDs[src_id];
                                        output.push_back(dst_id);
                                } else if (IDs[src_id] > IDs[dst_id]) {
                                        DEBUG("Decrease src's ID from %d to %d",
                                              IDs[src_id], IDs[dst_id]);
                                        IDs[src_id] = IDs[dst_id];
                                        output.push_back(src_id);
                                }
                        }
                }
                frontier = std::move(output);
                bool update = 1;
                while (update) {
                        update = 0;
                        for (int i = 0, e = edges.num_vertices; i < e; i++) {
                                int y = IDs[i];
                                int x = IDs[y];
                                DEBUG("%d has parent %d and grandparent %d",
                                      i, y, x);

                                if (x != y) {
                                        DEBUG("setting parent to grandparent");
                                        IDs[i] = x;
                                        update = 1;
                                }
                        }
                }
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

        // Validate against serial implementation
        int* SwarmIDs = IDs;
        IDs = new int32_t[edges.num_vertices];
        cc_serial();
        for (int i = 0, e = edges.num_vertices; i < e; i++) {
                assert(IDs[i] == SwarmIDs[i]);
        }
        printf("Successfully validated results!\n");
}
