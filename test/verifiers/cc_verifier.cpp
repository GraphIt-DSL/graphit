//
// Created by Yunming Zhang on 7/14/17.
//

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>
//#include "infra_gapbs/graph_verifier.h"
#include "intrinsics.h"
#include "verifier_utils.h"

using namespace std;


// Verifies CC result by performing a BFS from a vertex in each component
// - Asserts search does not reach a vertex with a different component label
// - If the graph is directed, it performs the search as if it was undirected
// - Asserts every vertex is visited (degree-0 vertex should have own label)
bool CCVerifier(const Graph &g, const pvector<NodeID> &comp) {
    unordered_map<NodeID, NodeID> label_to_source;
    for (NodeID n : g.vertices())
        label_to_source[comp[n]] = n;
    Bitmap visited(g.num_nodes());
    visited.reset();
    vector<NodeID> frontier;
    frontier.reserve(g.num_nodes());
    for (auto label_source_pair : label_to_source) {
        NodeID curr_label = label_source_pair.first;
        NodeID source = label_source_pair.second;
        frontier.clear();
        frontier.push_back(source);
        visited.set_bit(source);
        for (auto it = frontier.begin(); it != frontier.end(); it++) {
            NodeID u = *it;
            for (NodeID v : g.out_neigh(u)) {
                if (comp[v] != curr_label)
                    return false;
                if (!visited.get_bit(v)) {
                    visited.set_bit(v);
                    frontier.push_back(v);
                }
            }
            if (g.directed()) {
                for (NodeID v : g.in_neigh(u)) {
                    if (comp[v] != curr_label)
                        return false;
                    if (!visited.get_bit(v)) {
                        visited.set_bit(v);
                        frontier.push_back(v);
                    }
                }
            }
        }
    }
    for (NodeID n=0; n < g.num_nodes(); n++)
        if (!visited.get_bit(n))
            return false;
    return true;
}


int main(int argc, char* argv[]){
    std::cout << "running CC verifier " << std::endl;
    CLAppVerifier cli(argc, argv, "connected components");
    if (!cli.ParseArgs())
        return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();
    std::string verifier_input_filename = cli.verifier_input_results();
    pvector<int>* verifier_input_vector = readFileIntoVector<int>(verifier_input_filename);
    bool verification_flag = CCVerifier(g, *verifier_input_vector);
    if (verification_flag)
        std::cout << "CC verification SUCCESSFUL" << std::endl;
    else
        std::cout << "CC verification FAILED" << std::endl;

}
