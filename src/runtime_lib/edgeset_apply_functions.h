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

#include "infra_ligra/ligra/ligra.h"
#include "infra_ligra/ligra/utils.h"



template<typename APPLY_FUNC>
VertexSubset<NodeID> * edgeset_apply_push_parallel_sliding_queue_weighted_deduplicatied_from_vertexset_with_frontier (
        WGraph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {
    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    SlidingQueue<NodeID>* queue = from_vertexset->getSlidingQueue();
    next_frontier->sliding_queue_ = queue;
    queue->slide_window();

    if (g.get_flags_() == nullptr){
        //g.get_flags_() = new int[g.num_nodes()]();
        g.set_flags_( new int[g.num_nodes()]());
#pragma omp parallel for
        for (NodeID i = 0; i < g.num_nodes(); i++){
            g.get_flags_()[i] = 0;
        }
    }




#pragma omp parallel
    {
        QueueBuffer<NodeID> lqueue(*queue);
#pragma omp for
        for (auto q_iter = queue->shared_out_start; q_iter < queue->shared_out_end; q_iter++) {
            //since we now have wrap around, try to get the NodeId with mod
            NodeID src = * (queue->shared + (q_iter % queue->max_size));
            for (WNode dst : g.out_neigh(src)) {
                if ( apply_func(src, dst.v, dst.w) && CAS(&g.get_flags_()[dst.v], 0, 1)) {
                    lqueue.push_back(dst.v);
                }
            }
        }
        lqueue.flush();
    };

    next_frontier->num_vertices_ = queue->size();

    // reset the visited flags
#pragma omp parallel for
    for (auto q_iter = queue->shared_out_end; q_iter < queue->shared_in; q_iter++){
        NodeID node = * (queue->shared + (q_iter % queue->max_size));
        g.get_flags_()[node] = 0;
    }


    return next_frontier;
}

template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_push_parallel_sliding_queue_from_vertexset_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {
    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    SlidingQueue<NodeID>* queue = from_vertexset->getSlidingQueue();
    next_frontier->sliding_queue_ = from_vertexset->sliding_queue_;
    queue->slide_window();
    //std::cout << "queue size: " << queue->size() << std::endl;



#pragma omp parallel
    {
        QueueBuffer<NodeID> lqueue(*queue);
#pragma omp for
        for (auto q_iter = queue->begin(); q_iter < queue->end(); q_iter++) {
            NodeID src = *q_iter;
            for (NodeID dst : g.out_neigh(src)) {
                if (apply_func(src, dst)) {
                    lqueue.push_back(dst);
                }
            }
        }
        lqueue.flush();
    };

    //std::cout << "queue size: " << queue->size() << std::endl;
    next_frontier->num_vertices_ = queue->size();

    return next_frontier;
}

//template<typename APPLY_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_serial(Graph &g, APPLY_FUNC apply_func) {
//
//    for (NodeID u = 0; u < g.num_nodes(); u++) {
//        for (NodeID v : g.in_neigh(u))
//            apply_func(v, u);
//    }
//    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
//    //return nullptr;
//}

//template<typename APPLY_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_parallel(Graph &g, APPLY_FUNC apply_func) {
//
//    parallel_for (NodeID u = 0; u < g.num_nodes(); u++) {
//        for (NodeID v : g.in_neigh(u))
//            apply_func(v, u);
//    }
//    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
//}

//template<typename APPLY_FUNC>
//VertexSubset<NodeID> *edgeset_apply_push_serial(Graph &g, APPLY_FUNC apply_func) {
//
//    for (NodeID u = 0; u < g.num_nodes(); u++) {
//        for (NodeID v : g.in_neigh(u))
//            apply_func(u, v);
//    }
//    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
//}
//
//template<typename APPLY_FUNC, typename FROM_FUNC, typename TO_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_serial_from_filter_func_to_filter_func
//        (Graph &g, FROM_FUNC from_func, TO_FUNC to_func, APPLY_FUNC apply_func) {
//    for (NodeID u = 0; u < g.num_nodes(); u++) {
//        if (to_func(u)) {
//            for (NodeID v : g.in_neigh(u)) {
//                if (from_func(u)) {
//                    apply_func(v, u);
//                }
//            }
//        }
//    }
//    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
//};


//template<typename APPLY_FUNC, typename TO_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_serial_from_vertexset_to_filter_func_with_frontier
//        (Graph &g, VertexSubset<NodeID> *from_vertexset, TO_FUNC to_func, APPLY_FUNC apply_func) {
//
//    //The original GAPBS implmentation from BFS
//
//    //    int64_t awake_count = 0;
////    next.reset();
////#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
////    for (NodeID u=0; u < g.num_nodes(); u++) {
////        if (parent[u] < 0) {
////            for (NodeID v : g.in_neigh(u)) {
////                if (front.get_bit(v)) {
////                    parent[u] = v;
////                    awake_count++;
////                    next.set_bit(u);
////                    break;
////                }
////            }
////        }
//
//    Bitmap *next = new Bitmap(g.num_nodes());
//    Bitmap *current_frontier = from_vertexset->bitmap_;
//    int count = 0;
//
//    for (NodeID u = 0; u < g.num_nodes(); u++) {
//        if (to_func(u)) {
//            for (NodeID v : g.in_neigh(u)) {
//                if (current_frontier->get_bit(v)) {
//                    if (apply_func(v, u)) {
//                        next->set_bit(u);
//                        count++;
//                        if (!to_func(u)) break;
//                    }
//                }
//            }
//        }
//    }
//
//    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), count);
//    next_frontier->bitmap_ = next;
//
//    return next_frontier;
//};


//template<typename APPLY_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_serial_from_vertexset_with_frontier
//        (Graph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {
//
//    //The original GAPBS implmentation from BFS
//
//    //    int64_t awake_count = 0;
////    next.reset();
////#pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
////    for (NodeID u=0; u < g.num_nodes(); u++) {
////        if (parent[u] < 0) {
////            for (NodeID v : g.in_neigh(u)) {
////                if (front.get_bit(v)) {
////                    parent[u] = v;
////                    awake_count++;
////                    next.set_bit(u);
////                    break;
////                }
////            }
////        }
//
//    Bitmap *next = new Bitmap(g.num_nodes());
//    Bitmap *current_frontier = from_vertexset->bitmap_;
//    int count = 0;
//    for (NodeID u = 0; u < g.num_nodes(); u++) {
//        for (NodeID v : g.in_neigh(u)) {
//            if (current_frontier->get_bit(v)) {
//                if (apply_func(v, u)) {
//                    next->set_bit(u);
//                    count++;
//                }
//            }
//        }
//    }
//
//    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), count);
//    next_frontier->bitmap_ = next;
//
//    return next_frontier;
//};

//template<typename APPLY_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_serial_weighted_from_vertexset_with_frontier
//        (WGraph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {
//
//    Bitmap *next = new Bitmap(g.num_nodes());
//    Bitmap *current_frontier = from_vertexset->bitmap_;
//    int count = 0;
//    for (NodeID u = 0; u < g.num_nodes(); u++) {
//        //std::cout << "u : " << u << std::endl;
//        for (WNode s : g.in_neigh(u)) {
//            //std::cout << "s.v: " << s.v << " s.w: " << s.w << std::endl;
//            if (current_frontier->get_bit(s.v)) {
//                if (apply_func(s.v, u, s.w)) {
//                    next->set_bit(u);
//                    count++;
//                }
//            }
//        }
//    }
//
//    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), count);
//    next_frontier->bitmap_ = next;
//
//    return next_frontier;
//};


//template<typename APPLY_FUNC, typename FROM_FUNC, typename TO_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_serial_from_filter_func_to_filter_func_with_frontier
//        (Graph &g, FROM_FUNC from_func, TO_FUNC to_func, APPLY_FUNC apply_func) {
//
//
//    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
//    SlidingQueue<NodeID> queue = SlidingQueue<NodeID>(g.num_nodes());
//    for (int i = 0; i < g.num_nodes(); i++) {
//        queue.push_back(i);
//    }
//
//    {
//        QueueBuffer<NodeID> lqueue(queue);
//        for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
//            NodeID u = *q_iter;
//            for (NodeID v : g.out_neigh(u)) {
//                if (to_func(v)) {
//                    apply_func(u, v);
//                }
//            }
//        }
//        lqueue.flush();
//    }
//    //Here, we might be coping this too much
//    //next_frontier->dense_vertex_set_ = queue;
//    return next_frontier;
//
//}

/** DEPRECATED, start to use code gen
template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_pull_parallel(Graph &g, APPLY_FUNC apply_func) {

#pragma omp parallel for schedule(dynamic, 64)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
        for (NodeID v : g.in_neigh(u))
            apply_func(v, u);
    }
    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
}
 **/

//template<typename APPLY_FUNC>
//VertexSubset<NodeID> *edgeset_apply_pull_serial_weighted(WGraph &g, APPLY_FUNC apply_func) {
//
//#pragma omp parallel for schedule(dynamic, 64)
//    for (NodeID u = 0; u < g.num_nodes(); u++) {
//        for (WNode s : g.in_neigh(u))
//            apply_func(s.v, u, s.w);
//    }
//    return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
//}



/** Deprecated, using code gen now

template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_push_parallel_from_vertexset_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    long numVertices = g.num_nodes(), numEdges = g.num_edges();
    long m = from_vertexset->size();

    if (numVertices != from_vertexset->getVerticesRange()) {

        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    // used to generate nonzero indices to get degrees
    uintT *degrees = newA(uintT, m);
    NodeID *frontierVertices;

    // We probably need this when we get something that doesn't have a dense set, not sure
    // We can also write our own, the eixsting one doesn't quite work for bitvectors
    from_vertexset->toSparse();

    //from_vertexset->printDenseSet();

    frontierVertices = newA(NodeID, m);
    {
        parallel_for (long i = 0; i < m; i++) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
            frontierVertices[i] = v;
        }
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (outDegrees == 0) return next_frontier;

    uintT *offsets = degrees;
    long outEdgeCount = sequence::plusScan(offsets, degrees, m);
    uintE *outEdges = newA(uintE, outEdgeCount);

    {
        parallel_for (long i = 0; i < m; i++) {
            NodeID src = from_vertexset->dense_vertex_set_[i];
            uintT offset = offsets[i];
            //vertex vert = frontierVertices[i];
            //vert.decodeOutNghSparse(v, o, f, outEdges);
            int j = 0;
            for (NodeID dst : g.out_neigh(src)) {
                if (apply_func(src, dst)) {
                    outEdges[offset + j] = dst;
                } else {
                    outEdges[offset + j] = UINT_E_MAX;
                }
                j++;
            }

        }
    }
    uintE *nextIndices = newA(uintE, outEdgeCount);
    // Filter out the empty slots (marked with -1)
    long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
    free(outEdges);

    free(degrees);

    next_frontier->num_vertices_ = nextM;
    next_frontier->dense_vertex_set_ = nextIndices;

    return next_frontier;
}
**/


/** Deprecated, now using code generation for this schedule
template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_push_parallel_deduplicatied_from_vertexset_with_frontier
        (Graph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    long numVertices = g.num_nodes(), numEdges = g.num_edges();
    long m = from_vertexset->size();

    if (numVertices != from_vertexset->getVerticesRange()) {

        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    // used to generate nonzero indices to get degrees
    uintT *degrees = newA(uintT, m);
    NodeID *frontierVertices;

    // We probably need this when we get something that doesn't have a dense set, not sure
    // We can also write our own, the eixsting one doesn't quite work for bitvectors
    from_vertexset->toSparse();

    //from_vertexset->printDenseSet();

    frontierVertices = newA(NodeID, m);
    {
        for (long i = 0; i < m; i++) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
            frontierVertices[i] = v;
        }
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (outDegrees == 0) return next_frontier;

    uintT *offsets = degrees;
    long outEdgeCount = sequence::plusScan(offsets, degrees, m);
    uintE *outEdges = newA(uintE, outEdgeCount);

#ifdef TIME
    Timer edge_timer;
    edge_timer.Start();
#endif
    {
        parallel_for (long i = 0; i < m; i++) {
            NodeID src = from_vertexset->dense_vertex_set_[i];
            uintT offset = offsets[i];
            //vertex vert = frontierVertices[i];
            //vert.decodeOutNghSparse(v, o, f, outEdges);
            int j = 0;
            for (NodeID dst : g.out_neigh(src)) {
                if (apply_func(src, dst)) {
                    outEdges[offset + j] = dst;
                } else {
                    outEdges[offset + j] = UINT_E_MAX;
                }
                j++;
            }

        }
    }
#ifdef TIME
    edge_timer.Stop();
    PrintTime("Edge Apply Timer", edge_timer.Seconds());
#endif
    uintE *nextIndices = newA(uintE, outEdgeCount);
    //remove deuplications
    // remDuplicates(outEdges,flags,outEdgeCount,remDups);

    //using ligra's API for removing duplicates for now

#ifdef TIME
    Timer duplicate_timer;
    duplicate_timer.Start();
#endif

    remDuplicates(outEdges, NULL, outEdgeCount, g.num_nodes());

#ifdef TIME
    duplicate_timer.Stop();
    PrintTime("Deduplicate Time", duplicate_timer.Seconds());
#endif

    // Filter out the empty slots (marked with -1)
    long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
    free(outEdges);

    free(degrees);

    next_frontier->num_vertices_ = nextM;
    next_frontier->dense_vertex_set_ = nextIndices;

    return next_frontier;
}

 **/

// This is a push only case
//#define TIME
/** Deprecated, use code gen now
template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_push_parallel_weighted_deduplicatied_from_vertexset_with_frontier
        (WGraph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    long numVertices = g.num_nodes(), numEdges = g.num_edges();
    long m = from_vertexset->size();

    if (numVertices != from_vertexset->getVerticesRange()) {

        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    // used to generate nonzero indices to get degrees
    uintT *degrees = newA(uintT, m);

#ifdef TIME
    Timer out_d_timer;
    out_d_timer.Start();
#endif

    from_vertexset->toSparse();

    if (g.get_flags_() == nullptr)
        g.get_flags_() = new int[numVertices];

    parallel_for (int i = 0; i < numVertices; i++) {
        g.get_flags_()[i] = 0;
    }

    // We probably need this when we get something that doesn't have a dense set, not sure
    // We can also write our own, the eixsting one doesn't quite work for bitvectors
    from_vertexset->toSparse();

    //from_vertexset->printDenseSet();

    {
        parallel_for (long i = 0; i < m; i++) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
        }
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (outDegrees == 0) return next_frontier;

#ifdef TIME
    out_d_timer.Stop();
    PrintTime("Outdegree Time", out_d_timer.Seconds());
#endif

    uintT *offsets = degrees;
    long outEdgeCount = sequence::plusScan(offsets, degrees, m);
    uintE *outEdges = newA(uintE, outEdgeCount);


#ifdef TIME
    Timer apply_timer;
    apply_timer.Start();
#endif

    //#pragma omp parallel for schedule (dynamic)
    parallel_for (long i = 0; i < m; i++) {
        NodeID src = from_vertexset->dense_vertex_set_[i];
        uintT offset = offsets[i];
        //vertex vert = frontierVertices[i];
        //vert.decodeOutNghSparse(v, o, f, outEdges);
        int j = 0;
        for (WNode dst : g.out_neigh(src)) {
                //using CAS for deduplication
	  if (apply_func(src, dst.v, dst.w) && CAS(&(g.get_flags_()[dst.v]), 0, 1)) {
                    outEdges[offset + j] = dst.v;
            } else {
                outEdges[offset + j] = UINT_E_MAX;
            }
            j++;
        }

    }

#ifdef TIME
    apply_timer.Stop();
    PrintTime("Apply Time", apply_timer.Seconds());
#endif

    uintE *nextIndices = newA(uintE, outEdgeCount);
    //remove deuplications
    //remDuplicates(outEdges,flags,outEdgeCount,remDups);

    //using ligra's API for removing duplicates for now
#ifdef TIME
    Timer d_timer;
    d_timer.Start();
#endif
    //switch to use CAS version
    //remDuplicates(outEdges,NULL,outEdgeCount,g.num_nodes());

#ifdef TIME
    d_timer.Stop();
    PrintTime("Remove Duplicate Time", d_timer.Seconds());
#endif

    // Filter out the empty slots (marked with -1)
    long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
    free(outEdges);

    free(degrees);

    next_frontier->num_vertices_ = nextM;
    next_frontier->dense_vertex_set_ = nextIndices;

    return next_frontier;
}

 **/

/** DEPRECATED, we use code generation now
template<typename PULL_FUNC, typename PUSH_FUNC>
VertexSubset<NodeID> *edgeset_apply_hybrid_dense_parallel_deduplicatied_from_vertexset_with_frontier
        (Graph &g,
         VertexSubset<NodeID> *from_vertexset,
         PULL_FUNC pull_func,
         PUSH_FUNC push_func) {

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    long numVertices = g.num_nodes(), numEdges = g.num_edges();
    long m = from_vertexset->size();

    if (numVertices != from_vertexset->getVerticesRange()) {

        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    // used to generate nonzero indices to get degrees
    uintT *degrees = newA(uintT, m);

#ifdef TIME
    Timer out_d_timer;
    out_d_timer.Start();
#endif



    // We probably need this when we get something that doesn't have a dense set, not sure
    // We can also write our own, the eixsting one doesn't quite work for bitvectors
    from_vertexset->toSparse();

    //from_vertexset->printDenseSet();

    {
        parallel_for (long i = 0; i < m; i++) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
        }
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (outDegrees == 0) return next_frontier;

    if (m + outDegrees > numEdges / 20) {
        //do dense pull
        //std::cout << "edge apply dense" << std::endl;

        //read boolean array
        from_vertexset->toDense();
        free(degrees);

        //convert to bit vector
        // this would be an add on optimization (first match Ligra's performance)

        //Bitmap *next = new Bitmap(g.num_nodes());
        //Bitmap *current_frontier = from_vertexset->bitmap_;
        bool * next = newA(bool, g.num_nodes());
        parallel_for (int i = 0; i < numVertices; i++)next[i] = 0;

        parallel_for (NodeID u = 0; u < g.num_nodes(); u++) {
            //if (to_func(u)) {
                for (NodeID v : g.in_neigh(u)) {
                    //if (current_frontier->get_bit(v)) {
                    if (from_vertexset->bool_map_[v]) {
                        if (pull_func(v, u)) {
                            //write to a boolena array instead of bit vector
                            //next->set_bit(u);
                            next[u] = 1;
                            //if (!to_func(u)) break;
                        }
                    }
                }
            //}
        }

        // need to get a count out of the boolean array

        next_frontier->num_vertices_ = sequence::sum(next, numVertices);
        next_frontier->bool_map_ = next;
        return next_frontier;

    } else {
      //std::cout << "sparse" << std::endl;
      
        if (g.get_flags_() == nullptr)
	  g.get_flags_() = new int[numVertices]();

	parallel_for(long i = 0; i < m; i++){
	  g.get_flags_()[from_vertexset->dense_vertex_set_[i]] = 0;
	}
        //std::cout << "edge apply sparse" << std::endl;


        //do sparse push
        uintT *offsets = degrees;
        long outEdgeCount = sequence::plusScan(offsets, degrees, m);
        uintE *outEdges = newA(uintE, outEdgeCount);

#ifdef TIME
        Timer apply_timer;
    apply_timer.Start();
#endif
        //#pragma omp parallel for  schedule (dynamic, 1024)
        parallel_for (long i = 0; i < m; i++) {
            NodeID src = from_vertexset->dense_vertex_set_[i];
            uintT offset = offsets[i];
            //vertex vert = frontierVertices[i];
            //vert.decodeOutNghSparse(v, o, f, outEdges);
            int j = 0;
            for (NodeID dst : g.out_neigh(src)) {
                    //using CAS for deduplication, disabled for this library
	      if (push_func(src, dst) && CAS(&(g.get_flags_()[dst]), 0, 1)) {
                        outEdges[offset + j] = dst;
                    //outEdges[offset + j] = dst;
                } else {
                    outEdges[offset + j] = UINT_E_MAX;
                }
                j++;
            }
        }
#ifdef TIME
        apply_timer.Stop();
    PrintTime("Apply Time", apply_timer.Seconds());
#endif
        uintE *nextIndices = newA(uintE, outEdgeCount);
        long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
        free(outEdges);

        free(degrees);

        next_frontier->num_vertices_ = nextM;
        next_frontier->dense_vertex_set_ = nextIndices;


        return next_frontier;

    }
}
*/

/** Deprecated, now using code gen for this schedule
// no deduplication
template<typename TO_FUNC, typename PULL_FUNC, typename PUSH_FUNC>
VertexSubset<NodeID> *edgeset_apply_hybrid_dense_parallel_from_vertexset_to_filter_func_with_frontier
        (Graph &g,
         VertexSubset<NodeID> *from_vertexset,
         TO_FUNC to_func,
         PULL_FUNC pull_func,
         PUSH_FUNC push_func) {

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    long numVertices = g.num_nodes(), numEdges = g.num_edges();
    long m = from_vertexset->size();

    if (numVertices != from_vertexset->getVerticesRange()) {

        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    // used to generate nonzero indices to get degrees
    uintT *degrees = newA(uintT, m);

#ifdef TIME
    Timer out_d_timer;
    out_d_timer.Start();
#endif


    // We probably need this when we get something that doesn't have a dense set, not sure
    // We can also write our own, the eixsting one doesn't quite work for bitvectors
    from_vertexset->toSparse();

    //from_vertexset->printDenseSet();

    {
        parallel_for (long i = 0; i < m; i++) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
        }
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (outDegrees == 0) return next_frontier;

    if (m + outDegrees > numEdges / 20) {
        //do dense pull
        //std::cout << "edge apply dense" << std::endl;

        //read boolean array
        from_vertexset->toDense();
        free(degrees);

        //convert to bit vector
        // this would be an add on optimization (first match Ligra's performance)

        //Bitmap *next = new Bitmap(g.num_nodes());
        //Bitmap *current_frontier = from_vertexset->bitmap_;
        bool * next = newA(bool, g.num_nodes());
        parallel_for (int i = 0; i < numVertices; i++)next[i] = 0;

        int count = 0;

        parallel_for (NodeID u = 0; u < g.num_nodes(); u++) {
            if (to_func(u)) {
                for (NodeID v : g.in_neigh(u)) {
                    //if (current_frontier->get_bit(v)) {
                    if (from_vertexset->bool_map_[v]) {
                        if (pull_func(v, u)) {
                            //write to a boolena array instead of bit vector
                            //next->set_bit(u);
                            next[u] = 1;
                            if (!to_func(u)) break;
                        }
                    }
                }
            }
        }

        // need to get a count out of the boolean array

        next_frontier->num_vertices_ = sequence::sum(next, numVertices);
        next_frontier->bool_map_ = next;
        return next_frontier;

    } else {

        //std::cout << "edge apply sparse" << std::endl;


        //do sparse push
        uintT *offsets = degrees;
        long outEdgeCount = sequence::plusScan(offsets, degrees, m);
        uintE *outEdges = newA(uintE, outEdgeCount);

#ifdef TIME
        Timer apply_timer;
    apply_timer.Start();
#endif
        //#pragma omp parallel for  schedule (dynamic, 1024)
        parallel_for (long i = 0; i < m; i++) {
            NodeID src = from_vertexset->dense_vertex_set_[i];
            uintT offset = offsets[i];
            //vertex vert = frontierVertices[i];
            //vert.decodeOutNghSparse(v, o, f, outEdges);
            int j = 0;
            for (NodeID dst : g.out_neigh(src)) {
                if (push_func(src, dst)) {
                    //using CAS for deduplication, disabled for this library
//                    if (CAS(&(g.get_flags_()[dst]), 0, 1)) {
//                        outEdges[offset + j] = dst;
//                    }
                    outEdges[offset + j] = dst;
                } else {
                    outEdges[offset + j] = UINT_E_MAX;
                }
                j++;
            }
        }
#ifdef TIME
        apply_timer.Stop();
    PrintTime("Apply Time", apply_timer.Seconds());
#endif
        uintE *nextIndices = newA(uintE, outEdgeCount);
        long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
        free(outEdges);

        free(degrees);

        next_frontier->num_vertices_ = nextM;
        next_frontier->dense_vertex_set_ = nextIndices;

        return next_frontier;

    }
}

*/


/** DEPRECATED, use code gen now
template<typename APPLY_FUNC>
VertexSubset<NodeID> *edgeset_apply_hybrid_denseforward_parallel_weighted_deduplicatied_from_vertexset_with_frontier
        (WGraph &g, VertexSubset<NodeID> *from_vertexset, APPLY_FUNC apply_func) {

    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    long numVertices = g.num_nodes(), numEdges = g.num_edges();
    long m = from_vertexset->size();

    if (numVertices != from_vertexset->getVerticesRange()) {

        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    // used to generate nonzero indices to get degrees
    uintT *degrees = newA(uintT, m);

#ifdef TIME
    Timer out_d_timer;
    out_d_timer.Start();
#endif


    // We probably need this when we get something that doesn't have a dense set, not sure
    // We can also write our own, the eixsting one doesn't quite work for bitvectors
    from_vertexset->toSparse();

    //from_vertexset->printDenseSet();

    {
        parallel_for (long i = 0; i < m; i++) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
        }
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (outDegrees == 0) return next_frontier;
    if (m + outDegrees > numEdges / 20) {
      //std::cout << "edgemap dense forward" << std::endl;

        //ligra code
        //        bool* next = newA(bool,numVertices);
//        {parallel_for(long i=0;i<numVertices;i++) next[i] = 0;}
//        {parallel_for (long i=0; i<numVertices; i++){
//                if (vertexSubset[i]) {
//                    G[i].decodeOutNgh(i, vertexSubset, f, next);
//                }
//            }}

        from_vertexset->toDense();
        free(degrees);

//        Deprecated, switch to use boolean arrays due to performance issues
//        Bitmap * next = new Bitmap(g.num_nodes());
//        Bitmap * current = from_vertexset->bitmap_;
//        next->reset();

        bool *next = newA(bool, numVertices);
        parallel_for (int i = 0; i < numVertices; i++)next[i] = 0;
        bool *current = from_vertexset->bool_map_;

        //int64_t count = 0;
        //#pragma omp parallel for reduction(+ : count)  schedule (dynamic, 1024)
        parallel_for (NodeID u = 0; u < numVertices; u++) {
            //if (current->get_bit(u)){
            if (current[u]) {
	      for (WNode s : g.out_neigh(u)) {
                    if (apply_func(u, s.v, s.w)) {
                        //next->set_bit_atomic(s.v);
                        //count++;
                        next[s.v] = 1;
                    }
                }
            }
        }
        next_frontier->num_vertices_ = sequence::sum(next, numVertices);
        //next_frontier->bitmap_ = next;
        next_frontier->bool_map_ = next;
        return next_frontier;
    } else {


      if (g.get_flags_() == nullptr){
        g.get_flags_() = new int[numVertices]();
      }

        uintT *offsets = degrees;
        long outEdgeCount = sequence::plusScan(offsets, degrees, m);
        uintE *outEdges = newA(uintE, outEdgeCount);

#ifdef TIME
        Timer apply_timer;
    apply_timer.Start();
#endif
        //#pragma omp parallel for  schedule (dynamic, 1024)
        parallel_for (long i = 0; i < m; i++) {
            NodeID src = from_vertexset->dense_vertex_set_[i];
            uintT offset = offsets[i];
            //vertex vert = frontierVertices[i];
            //vert.decodeOutNghSparse(v, o, f, outEdges);
            int j = 0;
	    for (WNode dst : g.out_neigh(src)) {
	      if (apply_func(src, dst.v, dst.w) && CAS(&(g.get_flags_()[dst.v]), 0, 1)) {
                        outEdges[offset + j] = dst.v;

                } else {
                    outEdges[offset + j] = UINT_E_MAX;
                }
                j++;
            }
        }
#ifdef TIME
        apply_timer.Stop();
    PrintTime("Apply Time", apply_timer.Seconds());
#endif
        uintE *nextIndices = newA(uintE, outEdgeCount);
        long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
        free(outEdges);

        free(degrees);

        next_frontier->num_vertices_ = nextM;
        next_frontier->dense_vertex_set_ = nextIndices;
	for(int i = 0; i < nextM; i++){
	  g.get_flags_()[nextIndices[i]] = 0;
	}

        return next_frontier;
    }
}
*/

#endif //GRAPHIT_EDGESET_APPLY_FUNCTIONS_H
