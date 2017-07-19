//
// Created by Yunming Zhang on 7/11/17.
//

#ifndef GRAPHIT_EDGESET_APPLY_FUNCTIONS_H
#define GRAPHIT_EDGESET_APPLY_FUNCTIONS_H

#include "vertexsubset.h"
#include "infra_gapbs/builder.h"
#include "infra_gapbs/benchmark.h"
#include "infra_gapbs/bitmap.h"
#include "infra_gapbs/command_line.h"
#include "infra_gapbs/graph.h"
#include "infra_gapbs/platform_atomics.h"
#include "infra_gapbs/pvector.h"


template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial(Graph &g, APPLY_FUNC apply_func) {

    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (NodeID v : g.in_neigh(u))
            apply_func(v, u);
    }
    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
}

template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_push_serial(Graph &g, APPLY_FUNC apply_func) {

    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (NodeID v : g.in_neigh(u))
            apply_func(u, v);
    }
    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
}

template<typename APPLY_FUNC, typename FROM_FUNC, typename TO_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial_from_filter_func_to_filter_func
        (Graph &g, FROM_FUNC from_func, TO_FUNC to_func, APPLY_FUNC apply_func) {
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        if (to_func(u)){
            for (NodeID v : g.in_neigh(u)){
                if (from_func(u)){
                    apply_func(v, u);
                }
            }
        }
    }
    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
};


template<typename APPLY_FUNC, typename TO_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial_from_vertexset_to_filter_func_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, TO_FUNC to_func, APPLY_FUNC apply_func) {

    //The original GAPBS implmentation from BFS

    //    int64_t awake_count = 0;
//    next.reset();
//#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
//    for (NodeID u=0; u < g.num_nodes(); u++) {
//        if (parent[u] < 0) {
//            for (NodeID v : g.in_neigh(u)) {
//                if (front.get_bit(v)) {
//                    parent[u] = v;
//                    awake_count++;
//                    next.set_bit(u);
//                    break;
//                }
//            }
//        }

    Bitmap* next = new Bitmap(g.num_nodes());
    Bitmap* current_frontier = from_vertexset->bitmap_;
    int count = 0;

    for (NodeID u = 0; u < g.num_nodes(); u++) {
        if (to_func(u)) {
            for (NodeID v : g.in_neigh(u)) {
                if (current_frontier->get_bit(v)) {
                    if (apply_func(v, u)){
                        next->set_bit(u);
                        count++;
                    }
                }
            }
        }
    }

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), count);
    next_frontier->bitmap_ = next;

    return next_frontier;
};


template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial_from_vertexset_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {

    //The original GAPBS implmentation from BFS

    //    int64_t awake_count = 0;
//    next.reset();
//#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
//    for (NodeID u=0; u < g.num_nodes(); u++) {
//        if (parent[u] < 0) {
//            for (NodeID v : g.in_neigh(u)) {
//                if (front.get_bit(v)) {
//                    parent[u] = v;
//                    awake_count++;
//                    next.set_bit(u);
//                    break;
//                }
//            }
//        }

    Bitmap* next = new Bitmap(g.num_nodes());
    Bitmap* current_frontier = from_vertexset->bitmap_;
    int count = 0;
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (NodeID v : g.in_neigh(u)) {
            if (current_frontier->get_bit(v)) {
                if( apply_func(v, u)){
                    next->set_bit(u);
                    count++;
                }
            }
        }
    }

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), count);
    next_frontier->bitmap_ = next;

    return next_frontier;
};

template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial_weighted_from_vertexset_with_frontier
        (WGraph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {

    Bitmap* next = new Bitmap(g.num_nodes());
    Bitmap* current_frontier = from_vertexset->bitmap_;
    int count = 0;
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (WNode s : g.in_neigh(u)) {
            if (current_frontier->get_bit(s.v)) {
                if( apply_func(s.v, u, s.w)){
                    next->set_bit(u);
                    count++;
                }
            }
        }
    }

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), count);
    next_frontier->bitmap_ = next;

    return next_frontier;
};


template<typename APPLY_FUNC, typename FROM_FUNC, typename TO_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial_from_filter_func_to_filter_func_with_frontier
        (Graph &g, FROM_FUNC from_func, TO_FUNC to_func, APPLY_FUNC apply_func) {


    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    SlidingQueue<NodeID> queue = SlidingQueue<NodeID>(g.num_nodes());
    for (int i = 0; i < g.num_nodes(); i++){
        queue.push_back(i);
    }

    {
        QueueBuffer<NodeID> lqueue(queue);
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            for (NodeID v : g.out_neigh(u)) {
                if (to_func(v)) {
                    apply_func(u, v);
                }
            }
        }
        lqueue.flush();
    }
    //Here, we might be coping this too much
    next_frontier->dense_vertex_set_ = queue;
    return next_frontier;

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

template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_serial_weighted(WGraph &g, APPLY_FUNC apply_func) {

#pragma omp parallel for schedule(dynamic, 64)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (WNode s : g.in_neigh(u))
            apply_func(s.v, u, s.w);
    }
    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
}












//Code largely borrowed from TDStep for GAPBS
template<typename APPLY_FUNC, typename TO_FUNC>
VertexSubset<NodeID> *edgeset_apply_push_serial_from_vertexset_to_filter_func_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, TO_FUNC to_func, APPLY_FUNC apply_func) {


    //The original GAPBS implmentation from BFS

//    //need to get a SlidingQueue out of the vertexsubset
//    SlidingQueue<NodeID> &queue;
//
//    {
//        QueueBuffer<NodeID> lqueue(queue);
//        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
//            NodeID u = *q_iter;
//            for (NodeID v : g.out_neigh(u)) {
//                NodeID curr_val = parent[v];
////              Test, test and set
////                if (curr_val < 0) {
////                    if (compare_and_swap(parent[v], curr_val, u)) {
////                        lqueue.push_back(v);
////                    }
////                }
//            }
//        }
//        lqueue.flush();
//    }
//
    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    SlidingQueue<NodeID> &queue = from_vertexset->dense_vertex_set_;
    {
        QueueBuffer<NodeID> lqueue(queue);
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            for (NodeID v : g.out_neigh(u)) {
                if (to_func(v)) {
                    apply_func(u, v);
                }
            }
        }
        lqueue.flush();
    }
    //Here, we might be coping this too much
    next_frontier->dense_vertex_set_ = queue;
    return next_frontier;
}

template<typename APPLY_FUNC>
VertexSubset<NodeID> * edgeset_apply_push_parallel_deduplicatied_from_vertexset_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {
    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    SlidingQueue<NodeID> &queue = from_vertexset->dense_vertex_set_;
    {
        QueueBuffer<NodeID> lqueue(queue);
        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
            NodeID u = *q_iter;
            for (NodeID v : g.out_neigh(u)) {
                    if(apply_func(u, v)){
                        lqueue.push_back(v);
                    }            }
        }
        lqueue.flush();
    }
    //Here, we might be coping this too much
    next_frontier->dense_vertex_set_ = queue;
    return next_frontier;
}


#endif //GRAPHIT_EDGESET_APPLY_FUNCTIONS_H
