#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
int *num_paths;
float *dependences;
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
	swarm_runtime::sum_reduce(dependences[v], (1 / num_paths[v]));
}
void backward_update(int src, int dst) {
	swarm_runtime::sum_reduce(dependences[dst], dependences[src]);
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
	for(int trail = 0; trail < 1; trail++) {
		swarm_runtime::startTimer();
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
	swarm_runtime::clear_frontier(frontier);
		for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
			mark_unvisited(_iter);
		};
		swarm_runtime::builtin_retrieve(frontier_list, frontier);
		swarm_runtime::builtin_retrieve(frontier_list, frontier);
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
		float elapsed_time = swarm_runtime::stopTimer();
		swarm_runtime::print("elapsed time: ");
		swarm_runtime::print(elapsed_time);
		for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
			reset(_iter);
		};
	}
}
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	num_paths = new int[swarm_runtime::builtin_getVertices(edges)];
	dependences = new float[swarm_runtime::builtin_getVertices(edges)];
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
}