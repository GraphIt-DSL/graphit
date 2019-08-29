//
// Created by Yunming Zhang on 7/14/17.
//

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>
//#include "infra_gapbs/graph_verifier.h"
#include "intrinsics.h"
#include "verifier_utils.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
//const size_t kMaxBin = numeric_limits<size_t>::max()/2;


// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
    // Serial Dijkstra implementation to get oracle distances
    //pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
    pvector<WeightT> oracle_dist(g.num_nodes(), 2147483647);

    oracle_dist[source] = 0;
    typedef pair<WeightT, NodeID> WN;
    priority_queue<WN, vector<WN>, greater<WN>> mq;
    mq.push(make_pair(0, source));
    while (!mq.empty()) {
        WeightT td = mq.top().first;
        NodeID u = mq.top().second;
        mq.pop();
        if (td == oracle_dist[u]) {
            for (WNode wn : g.out_neigh(u)) {
                if (td + wn.w < oracle_dist[wn.v]) {
                    oracle_dist[wn.v] = td + wn.w;
                    mq.push(make_pair(td + wn.w, wn.v));
                }
            }
        }
    }
    // Report any mismatches
    bool all_ok = true;
    for (NodeID n : g.vertices()) {
        if (dist_to_test[n] != oracle_dist[n]) {
            cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
            all_ok = false;
        }
    }
    return all_ok;
}


int main(int argc, char* argv[]){
    std::cout << "running SSSP verifier " << std::endl;
    CLAppVerifier cli(argc, argv, "sssp");
    if (!cli.ParseArgs())
        return -1;
    WeightedBuilder b(cli);
    WGraph g = b.MakeGraph();
    std::string verifier_input_filename = cli.verifier_input_results();
    pvector<int>* verifier_input_vector = readFileIntoVector<int>(verifier_input_filename);
    NodeID starting_node = cli.start_vertex();
    bool verification_flag = SSSPVerifier(g, starting_node, *verifier_input_vector);
    if (verification_flag)
        std::cout << "SSSP verification SUCCESSFUL" << std::endl;
    else
        std::cout << "SSSP verification FAILED" << std::endl;

}
