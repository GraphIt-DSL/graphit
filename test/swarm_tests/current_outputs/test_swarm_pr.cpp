#include <tuple>
#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
double *old_rank;
double *new_rank;
int *out_degree;
double *contrib;
double *error;
int *generated_tmp_vector_2;
double damp;
double beta_score;
void error_generated_vector_op_apply_func_5(int v) {
	error[v] = 0;
}
void contrib_generated_vector_op_apply_func_4(int v) {
	contrib[v] = 0;
}
void generated_vector_op_apply_func_3(int v) {
	out_degree[v] = generated_tmp_vector_2[v];
}
void new_rank_generated_vector_op_apply_func_1(int v) {
	new_rank[v] = 0;
}
void old_rank_generated_vector_op_apply_func_0(int v) {
	old_rank[v] = (1 / swarm_runtime::builtin_getVertices(edges));
}
void computeContrib(int v) {
	contrib[v] = (old_rank[v] / out_degree[v]);
}
void updateEdge(int src, int dst) {
	swarm_runtime::sum_reduce(new_rank[dst], contrib[src]);
}
void updateVertex(int v) {
	double old_score = old_rank[v];
	new_rank[v] = (beta_score + (damp * new_rank[v]));
	error[v] = fabs((new_rank[v] - old_rank[v]));
	old_rank[v] = new_rank[v];
	new_rank[v] = 0;
}
void printRank(int v) {
	swarm_runtime::print(old_rank[v]);
}
void reset(int v) {
	old_rank[v] = (1 / swarm_runtime::builtin_getVertices(edges));
	new_rank[v] = 0;
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
	for(int trail = 0; trail < 10; trail++) {
		swarm_runtime::startTimer();
		for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
			reset(_iter);
		};
		for(int i = 0; i < 20; i++) {
			for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
				computeContrib(_iter);
			};
			for (int _iter = 0; _iter < edges.num_edges; _iter++) {
				int _src = edges.h_edge_src[_iter];
				int _dst = edges.h_edge_dst[_iter];
				updateEdge(_src, _dst);
			};
			for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
				updateVertex(_iter);
			};
		}
		double elapsed_time = swarm_runtime::stopTimer();
		swarm_runtime::print("elapsed time: ");
		swarm_runtime::print(elapsed_time);
	}
}
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	old_rank = new double[swarm_runtime::builtin_getVertices(edges)];
	new_rank = new double[swarm_runtime::builtin_getVertices(edges)];
	out_degree = new int[swarm_runtime::builtin_getVertices(edges)];
	contrib = new double[swarm_runtime::builtin_getVertices(edges)];
	error = new double[swarm_runtime::builtin_getVertices(edges)];
	generated_tmp_vector_2 = new int[swarm_runtime::builtin_getVertices(edges)];
	damp = 0.85;
	beta_score = ((1 - damp) / swarm_runtime::builtin_getVertices(edges));
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		old_rank_generated_vector_op_apply_func_0(_iter);
	};
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		new_rank_generated_vector_op_apply_func_1(_iter);
	};
	generated_tmp_vector_2 = swarm_runtime::builtin_getOutDegrees(edges);
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		generated_vector_op_apply_func_3(_iter);
	};
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		contrib_generated_vector_op_apply_func_4(_iter);
	};
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		error_generated_vector_op_apply_func_5(_iter);
	};
		SCC_PARALLEL( swarm_main(); );
}
