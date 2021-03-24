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
bool updateEdge(int src, int dst) {
	bool output1;
	parent[dst] = src;
	output1 = (bool)1;
	return output1;
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
	swarm::BucketQueue<int> swarm_frontier;
	swarm_runtime::VertexFrontier frontier = swarm_runtime::create_new_vertex_set(swarm_runtime::builtin_getVertices(edges), 0);
	int start_vertex = atoi(__argv[2]);
	swarm_runtime::builtin_addVertex(frontier, start_vertex);
	parent[start_vertex] = start_vertex;
	for (int i = 0; i < frontier.size(); i++){
		swarm_frontier.push_init(0, frontier[i]);
	}
	swarm_frontier.for_each_prio([](unsigned level, int src, auto push) {
		int32_t edgeZero = edges.h_src_offsets[src];
		int32_t edgeLast = edges.h_src_offsets[src+1];
		for (int i = edgeZero; i < edgeLast; i++) {
			int dst = edges.h_edge_dst[i];
			if (toFilter(dst)) {
				{
					bool output1;
					parent[dst] = src;
					output1 = (bool)1;
					if (output1) {
						push(level + 1, dst);
					}
				}
			}
		}
	}, [](unsigned level, int src) {
	});
	swarm_runtime::clear_frontier(frontier);
	deleteObject(frontier);
}


#include <unordered_set>
#include <cstdlib>

// This function is Victor's manual translation to C++ of the code here:
// https://github.com/GraphIt-DSL/graphit/blob/gpu_backend/apps/bfs.gt
static void bfs_serial(const int s) {
        std::vector<int> frontier = {s};
        parent[s] = s;
        while(frontier.size() != 0) {
                std::vector<int> output;
                std::unordered_set<int> output_set;
                for (int src : frontier) {
                        DEBUG("Push from %u", src);
                        int32_t edgeZero = edges.h_src_offsets[src];
                        int32_t edgeLast = edges.h_src_offsets[src+1];
                        for (int i = edgeZero; i < edgeLast; i++) {
                                int dst = edges.h_edge_dst[i];
                                DEBUG("Checking neighbor %d", dst);
                                if (parent[dst] == -1) {
                                        DEBUG("Visiting neighbor %d", dst);
                                        parent[dst] = src;
                                        if (!output_set.count(dst)) {
                                                output.push_back(dst);
                                                output_set.insert(dst);
                                        }
                                }
                        }
                }
                frontier = std::move(output);
        }
}

// For validating results
int level(int i, int* parent) {
        if (parent[i] == i) return 0;
        else return level(parent[i], parent);
}


int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	parent = new int[swarm_runtime::builtin_getVertices(edges)];
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		parent_generated_vector_op_apply_func_0(_iter);
	};
	SCC_PARALLEL( swarm_main(); );

	// Save a copy of Swarm's outputs before running the serial version
        auto swarm_parent = parent;
        parent = new int[edges.num_vertices];
        for (int i = 0, e = edges.num_vertices; i < e; i++)
                parent[i] = -1;
        bfs_serial(atoi(__argv[2]));

        // Compare Swarm's outputs with serial outputs
        for (int i = 0, e = edges.num_vertices; i < e; i++) {
                assert(level(i, swarm_parent) == level(i, parent));
        }
        printf("Successfully validated results!\n");


}
