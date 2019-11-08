//
// Created by Tugsbayasgalan Manlaibaatar on 2019-10-16.
//

#include <iostream>
#include <vector>
#include "intrinsics.h"
#include "verifier_utils.h"

using namespace std;
typedef float ScoreT;

bool TCVerifier(const Graph &g, size_t test_total) {
    size_t total = 0;
    vector<NodeID> intersection;
    intersection.reserve(g.num_nodes());
    for (NodeID u : g.vertices()) {
        for (NodeID v : g.out_neigh(u)) {
            auto new_end = set_intersection(g.out_neigh(u).begin(),
                                            g.out_neigh(u).end(),
                                            g.out_neigh(v).begin(),
                                            g.out_neigh(v).end(),
                                            intersection.begin());
            intersection.resize(new_end - intersection.begin());
            total += intersection.size();
        }
    }
    total = total / 6;  // each triangle was counted 6 times
    if (total != test_total)
        cout << total << " != " << test_total << endl;
    cout << total;
    return total == test_total;
}



int main(int argc, char* argv[]) {
    std::cout << "running TC verifier " << std::endl;
    CLAppVerifier cli(argc, argv, "triangular-counting");
    if (!cli.ParseArgs())
        return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();
    std::string verifier_input_filename = cli.verifier_input_results();
    size_t total = readFileIntoSize<size_t>(verifier_input_filename);
    if (g.directed()) {
        cout << "Input graph is directed but tc requires undirected" << endl;
        return -2;
    }

    bool verification_flag = TCVerifier(g, total);
    cout << "verification done" << endl;
    cout << verification_flag << endl;

    if (verification_flag)
        cout << "TC verification SUCCESSFUL" << endl;
    else
        cout << "TC verification FAILED" << endl;
};
