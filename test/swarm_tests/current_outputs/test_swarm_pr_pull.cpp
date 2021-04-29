#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
swarm_runtime::GraphT<int> transposed_edges;
double *old_rank;
double *new_rank;
int *out_degree;
double *contrib;
double *error;
int *generated_tmp_vector_2;
double damp;
double beta_score;

void error_generated_vector_op_apply_func_5(int v) {
	error[v] = ((float) 0) ;
}
void contrib_generated_vector_op_apply_func_4(int v) {
	contrib[v] = ((float) 0) ;
}
void generated_vector_op_apply_func_3(int v) {
	out_degree[v] = generated_tmp_vector_2[v];
}
void new_rank_generated_vector_op_apply_func_1(int v) {
	new_rank[v] = ((float) 0) ;
}
void old_rank_generated_vector_op_apply_func_0(int v) {
	old_rank[v] = (((float) 1)  / swarm_runtime::builtin_getVertices(edges));
}
void computeContrib(int v) {
	contrib[v] = (old_rank[v] / out_degree[v]);
}
void updateEdge(int src, int dst) {
	new_rank[dst] += contrib[src];
	//swarm_runtime::sum_reduce(new_rank[dst], contrib[src]);
}
void pullEdgeTraversal() {
	int n = swarm_runtime::builtin_getVertices(edges);
	for (int d = 0, m = n; d < m; d++) {  // iterate on dest vertices
		int32_t edgeZero = transposed_edges.h_src_offsets[d];
                int32_t edgeLast = transposed_edges.h_src_offsets[d+1];
		SCC_OPT_SERIAL_LOOP
                for (int j = edgeZero; j < edgeLast; j++) {  // grab backward edges from transposedEdges to get src's
			int s = transposed_edges.h_edge_dst[j];
			updateEdge(s, d); // call the applied func on src, dest
		}
	}
}
void updateVertex(int v) {
	double old_score = old_rank[v];
	new_rank[v] = (beta_score + (damp * new_rank[v]));
	error[v] = fabs((new_rank[v] - old_rank[v]));
	old_rank[v] = new_rank[v];
	new_rank[v] = ((float) 0) ;
}
void printRank(int v) {
	swarm_runtime::print(old_rank[v]);
}
void reset(int v) {
	old_rank[v] = (((float) 1)  / swarm_runtime::builtin_getVertices(edges));
	new_rank[v] = ((float) 0) ;
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		reset(_iter);
	};
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		computeContrib(_iter);
	};
	/*
	for (int _iter = 0, m = transposed_edges.num_edges; _iter < m; _iter++) {
		int _src = transposed_edges.h_edge_src[_iter];
		int _dst = transposed_edges.h_edge_dst[_iter];
		updateEdge(_src, _dst);
	};
	*/
	pullEdgeTraversal();
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		updateVertex(_iter);
	};
}

#include <iostream>
#include <fstream>
int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	transposed_edges = swarm_runtime::builtin_transpose(edges);
	old_rank = new double[swarm_runtime::builtin_getVertices(edges)];
	new_rank = new double[swarm_runtime::builtin_getVertices(edges)];
	out_degree = new int[swarm_runtime::builtin_getVertices(edges)];
	contrib = new double[swarm_runtime::builtin_getVertices(edges)];
	error = new double[swarm_runtime::builtin_getVertices(edges)];
	generated_tmp_vector_2 = new int[swarm_runtime::builtin_getVertices(edges)];
	damp = ((float) 0.85) ;
	beta_score = ((((float) 1)  - damp) / swarm_runtime::builtin_getVertices(edges));
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		old_rank_generated_vector_op_apply_func_0(_iter);
	};
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		new_rank_generated_vector_op_apply_func_1(_iter);
	};
	generated_tmp_vector_2 = swarm_runtime::builtin_getOutDegrees(edges);
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		generated_vector_op_apply_func_3(_iter);
	};
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		contrib_generated_vector_op_apply_func_4(_iter);
	};
	for (int _iter = 0, m = swarm_runtime::builtin_getVertices(edges); _iter < m; _iter++) {
		error_generated_vector_op_apply_func_5(_iter);
	};
	SCC_PARALLEL( swarm_main(); );

	std::ofstream f("pr_answers.txt");
        if (!f.is_open()) {
                printf("file open failed.\n");
                return -1;
        }
        for (int i = 0; i < swarm_runtime::builtin_getVertices(edges); i++) {
                f << out_degree[i] << std::endl;
        }
        f.close();	
}
