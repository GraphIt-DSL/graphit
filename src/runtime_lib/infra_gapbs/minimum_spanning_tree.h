//
// Created by Yunming Zhang on 5/19/19.
//

#ifndef GRAPHIT_MINIMUM_SPANNING_TREE_H
#define GRAPHIT_MINIMUM_SPANNING_TREE_H

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>

#include "builder.h"
#include "graph.h"
#include "benchmark.h"


//Takes as input a weighted graph, a starting point
//Returns a parent array

static NodeID * minimum_spanning_tree(WGraph g, NodeID start){
    // Serial Dijkstra implementation to get oracle distances
    //pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
    //pvector<WeightT> oracle_dist(g.num_nodes(), 2147483647);
    NodeID * parent = new NodeID[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        parent[i] = -1;
    }


    typedef pair<WeightT, pair<NodeID, NodeID > > WN;
    priority_queue<WN, vector<WN>, greater<WN>> mq;
    mq.push(make_pair(0, make_pair(start, start)));
    while (!mq.empty()) {
//        WeightT td = mq.top().first;
        NodeID u = mq.top().second.first;
        if (parent[u] == -1){parent[u] = mq.top().second.second; }
        mq.pop();
//        if (parent[u] == -1) { //Node is unvisited
            for (WNode wn : g.out_neigh(u)) {
                if (parent[wn.v] == -1) {
                //parent[wn.v] = u;
//            mq.push(make_pair(td + wn.w, wn.v));
                mq.push(make_pair(wn.w, make_pair(wn.v, u)));
                }
            }
//        }
    }
    return parent;
}


#endif //GRAPHIT_MINIMUM_SPANNING_TREE_H
