//
// Created by Xinyi Chen on 6/2/19.
//

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>
#include "infra_gapbs/graph_verifier.h"
#include "intrinsics.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;


// Serialized Dijkstra
int64_t GetShortestDistSum(const WGraph &g, NodeID source) {
    // Serial Dijkstra implementation to get oracel distances
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
  
  int64_t shortest_dist_sum = 0;
  for (NodeID n = 0; n < static_cast<NodeID>(oracle_dist.size()); n++) {
    if (oracle_dist[n] < kDistInf) {
      shortest_dist_sum += oracle_dist[n];
    }
  }
  return shortest_dist_sum;
}

