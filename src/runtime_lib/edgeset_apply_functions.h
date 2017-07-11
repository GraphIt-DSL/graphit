//
// Created by Yunming Zhang on 7/11/17.
//

#ifndef GRAPHIT_EDGESET_APPLY_FUNCTIONS_H
#define GRAPHIT_EDGESET_APPLY_FUNCTIONS_H

#include "vertexsubset.h"

template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial(Graph &g, APPLY_FUNC apply_func) {

    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (NodeID v : g.in_neigh(u))
            apply_func(v, u);
    }
    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
}


template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_parallel(Graph &g, APPLY_FUNC apply_func) {

#pragma omp parallel for schedule(dynamic, 64)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (NodeID v : g.in_neigh(u))
            apply_func(v, u);
    }
    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
}

template<typename APPLY_FUNC, typename TO_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial_from_vertexset_to_filter_func_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, TO_FUNC to_func) {


};


//Code largely borrowed from TDStep for GAPBS
template<typename APPLY_FUNC, typename TO_FUNC>
VertexSubset<NodeID> *edgeset_apply_push_serial_from_vertexset_to_filter_func_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, TO_FUNC to_func) {
    //need to get a SlidingQueue out of the vertexsubset
    SlidingQueue<NodeID> &queue;

#pragma omp parallel
    {
        QueueBuffer<NodeID> lqueue(queue);
#pragma omp for reduction(+ : scout_count)
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            for (NodeID v : g.out_neigh(u)) {
                NodeID curr_val = parent[v];
//              Test, test and set
//                if (curr_val < 0) {
//                    if (compare_and_swap(parent[v], curr_val, u)) {
//                        lqueue.push_back(v);
//                    }
//                }
            }
        }
        lqueue.flush();
    }

}

#endif //GRAPHIT_EDGESET_APPLY_FUNCTIONS_H
