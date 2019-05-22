//
// Created by kenny yang on 2019-01-22.
//

#include <iostream>
#include <vector>
//#include "infra_gapbs/graph_verifier.h"
#include "intrinsics.h"
#include "verifier_utils.h"

using namespace std;
typedef float ScoreT;

// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
bool BCVerifier(const Graph &g, NodeID source, NodeID num_iters,
                const pvector<ScoreT> &scores_to_test) {
    pvector<ScoreT> scores(g.num_nodes(), 0);
    for (int iter=0; iter < num_iters; iter++) {

        std::cout << "source node  " << source << std::endl;
        // BFS phase, only records depth & path_counts
        pvector<int> depths(g.num_nodes(), -1);
        depths[source] = 0;
        vector<NodeID> path_counts(g.num_nodes(), 0);
        path_counts[source] = 1;
        vector<NodeID> to_visit;
        to_visit.reserve(g.num_nodes());
        to_visit.push_back(source);
        for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
            NodeID u = *it;
            for (NodeID v : g.out_neigh(u)) {
                if (depths[v] == -1) {
                    depths[v] = depths[u] + 1;
                    to_visit.push_back(v);
                }
                if (depths[v] == depths[u] + 1)
                    path_counts[v] += path_counts[u];
            }
        }
        // Get lists of vertices at each depth
        vector<vector<NodeID>> verts_at_depth;
        for (NodeID n : g.vertices()) {
            if (depths[n] != -1) {
                if (depths[n] >= static_cast<int>(verts_at_depth.size()))
                    verts_at_depth.resize(depths[n] + 1);
                verts_at_depth[depths[n]].push_back(n);
            }
        }
        // Going from farthest to clostest, compute "depencies" (deltas)
        pvector<ScoreT> deltas(g.num_nodes(), 0);
        for (int depth=verts_at_depth.size()-1; depth >= 0; depth--) {
            for (NodeID u : verts_at_depth[depth]) {
                for (NodeID v : g.out_neigh(u)) {
                    if (depths[v] == depths[u] + 1) {
                        deltas[u] += static_cast<ScoreT>(path_counts[u]) /
                                     static_cast<ScoreT>(path_counts[v]) * (1 + deltas[v]);
                    }
                }
                scores[u] += deltas[u];
            }
        }
    }

    // Compare scores
    bool all_ok = true;
    for (NodeID n : g.vertices()) {
        if (abs(scores[n] - scores_to_test[n]) > 0.000001) {
            cout << n << ": " << scores[n] << " != " << scores_to_test[n] << endl;
            all_ok = false;
        }
    }
    return all_ok;
};




int main(int argc, char* argv[]) {
    std::cout << "running BC verifier " << std::endl;
    CLAppVerifier cli(argc, argv, "betweenness-centrality");
    if (!cli.ParseArgs())
        return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();
    SourcePicker<Graph> sp(g, cli.start_vertex());
    SourcePicker<Graph> vsp(g, cli.start_vertex());

    std::cout << "starting vertex  " << cli.start_vertex() << std::endl;

    std::string verifier_input_filename = cli.verifier_input_results();
    pvector<float>* verifier_input_vector = readFileIntoVector<float>(verifier_input_filename);
    bool verification_flag = BCVerifier(g, cli.start_vertex(), 1, *verifier_input_vector);
    cout << "verification done" << endl;
    cout << verification_flag << endl;


    if (verification_flag)
        cout << "BC verification SUCCESSFUL" << endl;
    else
        cout << "BC verification FAILED" << endl;





};
