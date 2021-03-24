#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
double *num_paths;
double *dependences;
bool *visited;
struct frontier_struct {
	int src;
	int frontier_insert_round_0;
};
void visited_generated_vector_op_apply_func_2(int v) {
	visited[v] = (bool)0;
}
void dependences_generated_vector_op_apply_func_1(int v) {
	dependences[v] = 0;
}
void num_paths_generated_vector_op_apply_func_0(int v) {
	num_paths[v] = 0;
}
bool forward_update(int src, int dst) {
	bool output4;
	bool num_paths_trackving_var_3 = (bool)0;
	num_paths_trackving_var_3 = swarm_runtime::sum_reduce(num_paths[dst], num_paths[src]);
	output4 = num_paths_trackving_var_3;
	return output4;
}
bool visited_vertex_filter(int v) {
	bool output;
	output = (visited[v]) == ((bool)0);
	return output;
}
void mark_visited(int v) {
	visited[v] = (bool)1;
}
void mark_unvisited(int v) {
	visited[v] = (bool)0;
}
void backward_vertex_f(int v) {
	visited[v] = (bool)1;
	//printf("%d old dep: %f\n", v, dependences[v]);
	swarm_runtime::sum_reduce(dependences[v], (1 / num_paths[v]));
	//printf("%d new dep: %f\n", v, dependences[v]);
}
void backward_update(int src, int dst) {
	//printf("%d old dep: %f\n", dst, dependences[dst]);
	swarm_runtime::sum_reduce(dependences[dst], dependences[src]);
	//printf("%d, new dep: %f\n", dst, dependences[dst]);
}
void final_vertex_f(int v) {
	if ((num_paths[v]) != (0)) {
		dependences[v] = ((dependences[v] - (1 / num_paths[v])) * num_paths[v]);
	} else {
		dependences[v] = 0;
	}
}
void reset(int v) {
	dependences[v] = 0;
	num_paths[v] = 0;
	visited[v] = 0;
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
	swarm_runtime::GraphT<int> transposed_edges = swarm_runtime::builtin_transpose(edges);
	swarm::BucketQueue<frontier_struct> swarm_frontier;
	swarm_runtime::VertexFrontier frontier = swarm_runtime::create_new_vertex_set(swarm_runtime::builtin_getVertices(edges), 0);
	int start_vertex = atoi(__argv[2]);
	swarm_runtime::builtin_addVertex(frontier, start_vertex);
	num_paths[start_vertex] = 1;
	visited[start_vertex] = (bool)1;
	int round = 0;
	swarm_runtime::VertexFrontierList frontier_list = swarm_runtime::create_new_vertex_frontier_list(swarm_runtime::builtin_getVertices(edges));
	swarm_runtime::builtin_insert(frontier_list, frontier);
	for (int i = 0; i < frontier.size(); i++){
		swarm_frontier.push_init(0, frontier_struct{frontier[i], swarm_runtime::builtin_get_size(frontier_list)});
	}
	swarm_frontier.for_each_prio([&round, &frontier_list](unsigned level, frontier_struct src_struct, auto push) {
		switch (level % 4) {
		case 0: {
			push(level + 1, frontier_struct{src_struct.src, src_struct.frontier_insert_round_0});
			break;
		}
		case 1: {
			int32_t edgeZero = edges.h_src_offsets[src_struct.src];
			int32_t edgeLast = edges.h_src_offsets[src_struct.src+1];
			for (int i = edgeZero; i < edgeLast; i++) {
				int dst = edges.h_edge_dst[i];
				int src = src_struct.src;
				if (visited_vertex_filter(dst)) {
					{
						bool output4;
						bool num_paths_trackving_var_3 = (bool)0;
						num_paths_trackving_var_3 = swarm_runtime::sum_reduce(num_paths[dst], num_paths[src]);
						output4 = num_paths_trackving_var_3;
						if (output4) {
							push(level + 1, frontier_struct{dst, src_struct.frontier_insert_round_0});
						}
					}
				}
			}
			break;
		}
		case 2: {
			mark_visited(src_struct.src);
;
			push(level + 1, frontier_struct{src_struct.src, src_struct.frontier_insert_round_0});
			break;
		}
		case 3: {
			swarm_runtime::builtin_insert(frontier_list, src_struct.src, src_struct.frontier_insert_round_0);
			push(level + 1, frontier_struct{src_struct.src, src_struct.frontier_insert_round_0 + 1});
			break;
		}
		}
	}, [&round, &frontier_list](unsigned level, frontier_struct src_struct) {
		switch (level % 4) {
		case 0: {
			round = (round + 1);
			break;
		}
		case 1: {
			break;
		}
		case 2: {
			break;
		}
		case 3: {
			swarm_runtime::builtin_update_size(frontier_list, src_struct.frontier_insert_round_0);
			break;
		}
	}
});

//printf("Final round n swarm: %d\n", round);
swarm_runtime::clear_frontier(frontier);
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		mark_unvisited(_iter);
	};
	swarm_runtime::builtin_retrieve(frontier_list, frontier);
	
	// We don't have the empty frontier anymore
	// swarm_runtime::builtin_retrieve(frontier_list, frontier);
	
	for (int i = 0; i < frontier.size(); i++) {
		int32_t current = frontier[i];
		backward_vertex_f(current);
	};
	round = (round - 1);
	while ((round) > (0)) {
		for (int i = 0; i < frontier.size(); i++) {
			int32_t current = frontier[i];
			int32_t edgeZero = transposed_edges.h_src_offsets[current];
			int32_t edgeLast = transposed_edges.h_src_offsets[current+1];
			for (int j = edgeZero; j < edgeLast; j++) {
				int ngh = transposed_edges.h_edge_dst[j];
				if (visited_vertex_filter(ngh)) {
					backward_update(current, ngh);
				}
			}
		};
		swarm_runtime::builtin_retrieve(frontier_list, frontier);
		//printf("frontier size for backward vertex f: %d round: %d\n", frontier.size(), round);
		for (int i = 0; i < frontier.size(); i++) {
			int32_t current = frontier[i];
			backward_vertex_f(current);
		};
		round = (round - 1);
	}
	deleteObject(frontier);
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		final_vertex_f(_iter);
	};
}

#include <vector>
#include <unordered_set>
#include <cstdlib>

// This function is Victor's manual translation to C++ of the code here:
// https://github.com/GraphIt-DSL/graphit/blob/gpu_backend/apps/bc.gt
static void bc_serial(const int s) {
        std::vector<int> frontier = {s};
        num_paths[s] = 1;
        visited[s] = true;
        int round = 0;
        std::vector<std::vector<int>> frontier_list;
        frontier_list.push_back(frontier);

        // forward pass to propagate num_paths
        while(frontier.size() != 0) {
                round++;
                std::vector<int> output;
                std::unordered_set<int> output_set;
                for (int current : frontier) {
                        DEBUG("Forward from %u (round %u)", current, round);
                        int32_t edgeZero = edges.h_src_offsets[current];
                        int32_t edgeLast = edges.h_src_offsets[current+1];
                        for (int i = edgeZero; i < edgeLast; i++) {
                                int ngh = edges.h_edge_dst[i];
                                DEBUG("Checking neighbor %d", ngh);
                                if (!visited[ngh]) {
                                        //printf("Increasing neighbor's paths from %f to %f (+%f)",
                                        //     num_paths[ngh],
                                        //      num_paths[ngh] + num_paths[current],
                                        //      num_paths[current]);
                                        num_paths[ngh] += num_paths[current];
                                        if (!output_set.count(ngh)) {
                                                output.push_back(ngh);
                                                output_set.insert(ngh);
                                        }
                                }
                        }
                }
                for (int ngh : output) {
                        //printf("Marking %u visited (round %u)", ngh, round);
                        visited[ngh] = true;
                }
                frontier_list.push_back(output);
                frontier = std::move(output);
        }
	//printf("regular: %d rounds\n", round);
        DEBUG("Finished forward pass after %d rounds", round);
        // We need a transposed graph here, but for now just use a symmetric graph for correctness

        // Resetting the visited information for the backward pass
        for (int i = 0, e = edges.num_vertices; i < e; i++) {
                visited[i] = false;
        }

        // pop off the empty frontier
        assert(frontier_list.back().empty());
        frontier_list.pop_back();

        DEBUG("Running backward pass...");

        // start from the last frontier
        frontier = std::move(frontier_list.back()); frontier_list.pop_back();
        for (int v : frontier) {
                DEBUG("Marking %d visited, increasing dependences from %e to %e (+%e)",
                      v, dependences[v], dependences[v] + 1.0/num_paths[v], 1.0/num_paths[v]);
                visited[v] = true;
                dependences[v] += 1.0 / num_paths[v];
        }
        round--;
// backward pass to accumulate the dependencies
        while (round) {
                DEBUG("Backwards round %d", round);
                for (int current : frontier) {
                        DEBUG("Backwards from %d", current);
                        int32_t edgeZero = edges.h_src_offsets[current];
                        int32_t edgeLast = edges.h_src_offsets[current+1];
                        for (int i = edgeZero; i < edgeLast; i++) {
                                int ngh = edges.h_edge_dst[i];
                                DEBUG("Checking neighbor %d", ngh);
                                if (!visited[ngh]) {
                                        //printf("Increasing dependences from %e to %e (+%e)",
                                        //      dependences[ngh],
                                        //      dependences[ngh] + dependences[current],
                                        //      dependences[current]);
                                        dependences[ngh] += dependences[current];
                                }
                        }
                }
                frontier = std::move(frontier_list.back());
                frontier_list.pop_back();
                for (int v : frontier) {
                        //printf("Marking %d visited, increasing dependences from %e to %e (+%e)",
                        //      v, dependences[v], dependences[v] + 1.0/num_paths[v], 1.0/num_paths[v]);
                        visited[v] = true;
                        dependences[v] += 1.0 / num_paths[v];
                }
                round--;
        }

        for (int i = 0, e = edges.num_vertices; i < e; i++) {
                if (num_paths[i])
                        dependences[i] = (dependences[i] - 1.0 / num_paths[i]) * num_paths[i];
                else
                        dependences[i] = 0;
        }
}

int main(int argc, char* argv[]) {
        __argc = argc;
        __argv = argv;
        swarm_runtime::load_graph(edges, __argv[1]);
        num_paths = new double[swarm_runtime::builtin_getVertices(edges)];
        dependences = new double[swarm_runtime::builtin_getVertices(edges)];
        visited = new bool[swarm_runtime::builtin_getVertices(edges)];
        for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
                num_paths_generated_vector_op_apply_func_0(_iter);
        };
        for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
                dependences_generated_vector_op_apply_func_1(_iter);
        };
        for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
                visited_generated_vector_op_apply_func_2(_iter);
        };
        SCC_PARALLEL( swarm_main(); );

        auto swarm_visited = visited;
        auto swarm_num_paths = num_paths;
        auto swarm_dependences = dependences;
        visited = new bool[edges.num_vertices]();
        num_paths = new double[edges.num_vertices]();
        dependences = new double[edges.num_vertices]();
        bc_serial(atoi(__argv[2]));
	/*
        for (int i = 0; i < edges.num_vertices;i++) {
		std::cout << i << " " << swarm_visited[i] << " " << visited[i] << std::endl;
	}
        for (int i = 0; i < edges.num_vertices;i++) {
		std::cout << i << " " << swarm_num_paths[i] << " " << num_paths[i] << std::endl;
	}
        for (int i = 0; i < edges.num_vertices;i++) {
		std::cout << i << " " << swarm_dependences[i] << " " << dependences[i] << std::endl;
	}
	*/
	// Compare Swarm's outputs with serial outputs
        for (int i = 0, e = edges.num_vertices; i < e; i++) {
                assert(swarm_visited[i] == visited[i]);
                assert(std::abs(swarm_num_paths[i] - num_paths[i]) <=
                               0.0001 * std::abs(num_paths[i]));
                assert(std::abs(swarm_dependences[i] - dependences[i]) <=
                               0.0001 * std::abs(dependences[i]));
        }
        printf("Successfully validated results!\n");


}
