//
// Created by Yunming Zhang on 7/12/17.
//


#include <iostream>
#include <vector>
//#include "infra_gapbs/graph_verifier.h"
#include "intrinsics.h"
#include "verifier_utils.h"

using namespace std;

// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
    pvector<int> depth(g.num_nodes(), -1);
    depth[source] = 0;
    vector<NodeID> to_visit;
    to_visit.reserve(g.num_nodes());
    to_visit.push_back(source);
    for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
        NodeID u = *it;
        for (NodeID v : g.out_neigh(u)) {
            if (depth[v] == -1) {
                depth[v] = depth[u] + 1;
                to_visit.push_back(v);
            }
        }
    }
    for (NodeID u : g.vertices()) {
        if ((depth[u] != -1) && (parent[u] != -1)) {
            if (u == source) {
                if (!((parent[u] == u) && (depth[u] == 0))) {
                    cout << "Source wrong" << endl;
                    return false;
                }
                continue;
            }
            bool parent_found = false;
            for (NodeID v : g.in_neigh(u)) {
                if (v == parent[u]) {
                    if (depth[v] != depth[u] - 1) {
                        cout << "Wrong depths for " << u << " & " << v << endl;
                        return false;
                    }
                    parent_found = true;
                    break;
                }
            }
            if (!parent_found) {
                cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
                return false;
            }
        } else if (depth[u] != parent[u]) {
            cout << "Reachability mismatch" << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]){
    std::cout << "running BFS verifier " << std::endl;
    CLAppVerifier cli(argc, argv, "breadth-first search");
    if (!cli.ParseArgs())
        return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();
    std::string verifier_input_filename = cli.verifier_input_results();
    pvector<int>* verifier_input_vector = readFileIntoVector<int>(verifier_input_filename);
    NodeID starting_node = cli.start_vertex();
    bool verification_flag = BFSVerifier(g, starting_node, *verifier_input_vector);
    if (verification_flag)
        std::cout << "BFS verification SUCCESSFUL" << std::endl;
    else
        std::cout << "BFS verification FAILED" << std::endl;

}
