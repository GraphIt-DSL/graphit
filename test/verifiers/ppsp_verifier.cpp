//
// Created by Yunming Zhang on 7/14/17.
//

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>
#include "intrinsics.h"
#include "verifier_utils.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
//const size_t kMaxBin = numeric_limits<size_t>::max()/2;


// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source, NodeID dest,
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


    // Report mismatch on the distance to the destination node

    bool all_ok = false;
    if (dist_to_test[dest] == oracle_dist[dest]) all_ok = true;
    else {
        cout << "measured dist: " << dist_to_test[dest] << endl;
        cout << "oracle dist: " << oracle_dist[dest] << endl;
    }

    return all_ok;

}


int main(int argc, char* argv[]){
    std::cout << "running PPSP verifier " << std::endl;
    CLAppVerifier cli(argc, argv, "ppsp");
    if (!cli.ParseArgs())
        return -1;
    WeightedBuilder b(cli);
    WGraph g = b.MakeGraph();
    std::string verifier_input_filename = cli.verifier_input_results();
    pvector<int>* verifier_input_vector = readFileIntoVector<int>(verifier_input_filename);
    NodeID starting_node = cli.start_vertex();
    NodeID ending_node = cli.end_vertex();

    bool verification_flag = SSSPVerifier(g, starting_node, ending_node, *verifier_input_vector);
    if (verification_flag)
        std::cout << "PPSP verification SUCCESSFUL" << std::endl;
    else
        std::cout << "PPSP verification FAILED" << std::endl;

}
