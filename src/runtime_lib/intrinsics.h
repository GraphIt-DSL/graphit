//
// Created by Yunming Zhang on 4/25/17.
//

#ifndef GRAPHIT_INTRINSICS_H_H
#define GRAPHIT_INTRINSICS_H_H



/* Julinne requirements -- should be in this order only */
#include <algorithm>

#ifdef CILK
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#endif
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <limits>

#if !defined __APPLE__ && !defined LOWMEM
#include <malloc.h>
#endif

#include <math.h>

#if defined(OPENMP)
#include <omp.h>
#endif

#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <unistd.h>


#define ulong unsigned long
namespace julienne {
#include "infra_julienne/priority_queue.h"
#include "infra_julienne/parallel.h"
template <typename X, typename Y>
struct EdgeMap;
#include "infra_julienne/graph.h"
}
namespace julienne {
}

static julienne::graph<julienne::symmetricVertex> __julienne_null_graph(NULL, 0, 0, NULL);

#undef INT_T_MAX
#undef UINT_T_MAX

/* Julienne requirements end */


#include <vector>
#include "infra_gapbs/builder.h"
#include "infra_gapbs/benchmark.h"
#include "infra_gapbs/intersections.h"
#include "infra_gapbs/bitmap.h"
#include "infra_gapbs/command_line.h"
#include "infra_gapbs/graph.h"
#include "infra_gapbs/platform_atomics.h"
#include "infra_gapbs/pvector.h"
#include "infra_gapbs/eager_priority_queue.h"
#include <queue>
#include "infra_gapbs/timer.h"
#include "infra_gapbs/sliding_queue.h"
#include "infra_gapbs/ordered_processing.h"

#include "edgeset_apply_functions.h"
#include <unordered_map>
#include <unordered_set>

#include "infra_ligra/ligra/ligra.h"

#include "vertexsubset.h"

namespace julienne {
#include "infra_julienne/IO.h"
#include "infra_julienne/edgeMapReduce.h"
}

#include <time.h>
#include <chrono>
#include "infra_gapbs/minimum_spanning_tree.h"
#include <float.h>


template <typename T>
static T builtin_sum(T* input_vector, int num_elem){
    //Serial Code for summation
    //T output_sum = 0;
    //for (int i = 0; i < num_elem; i++){
    //    output_sum += input_vector[i];
    //}

    T reduce_sum = sequence::plusReduce(input_vector, num_elem);

    return reduce_sum;
}

template <typename T>
static T builtin_max(T* input_vector, int num_elem){

    T reduce_max = sequence::maxReduce(input_vector, num_elem);

    return reduce_max;
}


static int max(double val1, int val2){
    return max(int(val1), val2);
}

static bool writeMin(int * val_array, int index, int new_val){
    return writeMin(&val_array[index], new_val);
}

//For now, assume the weights are ints, this would be good enough for now
// Later, we can change the parser, to supply type information to the library call
static WGraph builtin_loadWeightedEdgesFromFile(std::string file_name){
    CLBase cli (file_name);
    WeightedBuilder weighted_builder (cli);
    WGraph g = weighted_builder.MakeGraph();
    return g;
}

static Graph builtin_loadEdgesFromFile(std::string file_name){
    CLBase cli (file_name);
    Builder builder (cli);
    Graph g = builder.MakeGraph();
    return g;
}


static Graph builtin_loadEdgesFromCSR(const int32_t* indptr, const NodeID* indices, int num_nodes, int num_edges) {

    typedef EdgePair<NodeID, NodeID> Edge;
    typedef pvector<Edge> EdgeList;
    typedef pvector<int32_t> DegreeList;
    EdgeList el;
    el.resize(num_edges);
    DegreeList dl;
    dl.resize(num_nodes);

    #pragma omp parallel for schedule(dynamic, 64)
    for (NodeID x = 0; x < num_nodes; x++) {
        int32_t degree = indptr[x+1] - indptr[x];
        dl[x] = degree;
    }

    CLBase cli(0, NULL);
    BuilderBase<NodeID> bb(cli);
    auto prefSum = bb.ParallelPrefixSum(dl);

    #pragma omp parallel for schedule(dynamic, 64)
    for (NodeID x = 0; x < num_nodes; x++) {
        auto startOffset = prefSum[x];
        for(int32_t i = 0; i < dl[x]; i++) {
            el[startOffset+i] = Edge(x, indices[startOffset + i]);
        }
    }

    return bb.MakeGraphFromEL(el);
}
static WGraph builtin_loadWeightedEdgesFromCSR(const int32_t *data, const int32_t *indptr, const NodeID *indices, int num_nodes, int num_edges) {
	typedef EdgePair<NodeID, WNode> Edge;
	typedef pvector<Edge> EdgeList;
    typedef pvector<int32_t> DegreeList;
	EdgeList el;
	el.resize(num_edges);
	DegreeList dl;
	dl.resize(num_nodes);

    #pragma omp parallel for schedule(dynamic, 64)
    for (NodeID x = 0; x < num_nodes; x++) {
        int32_t degree = indptr[x+1] - indptr[x];
        dl[x] = degree;
    }

    CLBase cli(0, NULL);
    BuilderBase<NodeID, WNode, WeightT> bb(cli);
    auto prefSum = bb.ParallelPrefixSum(dl);
    bb.needs_weights_ = false;

    #pragma omp parallel for schedule(dynamic, 64)
    for (NodeID x = 0; x < num_nodes; x++) {
        auto startOffset = prefSum[x];
        for(int32_t i = 0; i < dl[x]; i++) {
            el[startOffset+i] = Edge(x, NodeWeight<NodeID, WeightT>(indices[startOffset + i], data[startOffset + i]));
        }
    }

	return bb.MakeGraphFromEL(el);	
}

static int builtin_getVertices(Graph &edges){
    return edges.num_nodes();
}

static int builtin_getVertices(WGraph &edges){
    return edges.num_nodes();
}

static NodeID builtin_getOutDegree(Graph &edges, NodeID src){
    return edges.out_degree(src);
}

static NodeID builtin_getOutDegree(WGraph &edges, NodeID src){
    return edges.out_degree(src);
}

static Graph builtin_relabel(Graph &edges) {

    // GAPBS way to figure out if the graph is worth relabelling
    auto worthLabelling = [](Graph &g) {
        int64_t average_degree = g.num_edges() / g.num_nodes();
        if (average_degree < 10)
            return false;
        SourcePicker<Graph> sp(g);
        int64_t num_samples = min(int64_t(1000), g.num_nodes());
        int64_t sample_total = 0;
        pvector<int64_t> samples(num_samples);
        for (int64_t trial=0; trial < num_samples; trial++) {
            samples[trial] = g.out_degree(sp.PickNext());
            sample_total += samples[trial];
        }
        sort(samples.begin(), samples.end());
        double sample_average = static_cast<double>(sample_total) / num_samples;
        double sample_median = samples[num_samples/2];
        return sample_average / 1.3 > sample_median;
    };

    if (worthLabelling(edges)) {
        Graph relabeledGraph = Builder::RelabelByDegree(edges);
        return relabeledGraph;
    }

    return edges;
}

static VertexSubset<NodeID>* builtin_getNgh(Graph &edges, NodeID src){
    auto v =  new VertexSubset<NodeID>(edges.out_degree(src));
    v->dense_vertex_set_ = (unsigned int*) edges.out_neigh(src).begin();
    return v;
}

static VertexSubset<NodeID>* builtin_getNgh(WGraph &edges, NodeID src){
    auto v =  new VertexSubset<NodeID>(edges.out_degree(src));
    v->dense_vertex_set_ = (unsigned int*) edges.out_neigh(src).begin();
    return v;
}


static size_t hiroshiVertexIntersection(VertexSubset<NodeID>* A, VertexSubset<NodeID>* B, size_t totalA, size_t totalB, NodeID dest) {
    return intersectSortedNodeSetHiroshi((NodeID *) A->dense_vertex_set_, (NodeID *) B->dense_vertex_set_, totalA, totalB, dest);

}

static size_t multiSkipVertexIntersection(VertexSubset<NodeID>* A, VertexSubset<NodeID>* B, size_t totalA, size_t totalB, NodeID dest) {
    return intersectSortedNodeSetMultipleSkip((NodeID *) A->dense_vertex_set_, (NodeID *) B->dense_vertex_set_, totalA, totalB, dest);

}

static size_t naiveVertexIntersection(VertexSubset<NodeID>* A, VertexSubset<NodeID>* B, size_t totalA, size_t totalB, NodeID dest) {
    return intersectSortedNodeSetNaive((NodeID *) A->dense_vertex_set_, (NodeID *) B->dense_vertex_set_, totalA, totalB, dest);
}

static size_t combinedVertexIntersection(VertexSubset<NodeID>* A, VertexSubset<NodeID>* B, size_t totalA, size_t totalB) {
    // currently just has fixed thresholds
    return intersectSortedNodeSetCombined((NodeID *) A->dense_vertex_set_, (NodeID *) B->dense_vertex_set_, totalA, totalB, 1000, 0.1);
}

static size_t binarySearchIntersection(VertexSubset<NodeID>* A, VertexSubset<NodeID>* B, size_t totalA, size_t totalB) {
    return intersectSortedNodeSetBinarySearch((NodeID *) A->dense_vertex_set_, (NodeID *) B->dense_vertex_set_, totalA, totalB);
}

static size_t hiroshiVertexIntersectionNeighbor(Graph &edges, NodeID src, NodeID dest) {
    auto iter_src = edges.out_neigh(src).begin();
    auto iter_dest = edges.out_neigh(dest).begin();
    auto srcTotal = edges.out_degree(src);
    auto destTotal = edges.out_degree(dest);

    return intersectSortedNodeSetHiroshi(iter_src, iter_dest, srcTotal, destTotal, dest);

}

static size_t multiSkipVertexIntersectionNeighbor(Graph &edges, NodeID src, NodeID dest) {
    auto iter_src = edges.out_neigh(src).begin();
    auto iter_dest = edges.out_neigh(dest).begin();
    auto srcTotal = edges.out_degree(src);
    auto destTotal = edges.out_degree(dest);

    return intersectSortedNodeSetMultipleSkip(iter_src, iter_dest, srcTotal, destTotal, dest);

}

static size_t naiveVertexIntersectionNeighbor(Graph &edges, NodeID src, NodeID dest) {
    auto iter_src = edges.out_neigh(src).begin();
    auto iter_dest = edges.out_neigh(dest).begin();
    auto srcTotal = edges.out_degree(src);
    auto destTotal = edges.out_degree(dest);

    return intersectSortedNodeSetNaive(iter_src, iter_dest, srcTotal, destTotal, dest);

}

static size_t combinedVertexIntersectionNeighbor(Graph &edges, NodeID src, NodeID dest) {
    auto iter_src = edges.out_neigh(src).begin();
    auto iter_dest = edges.out_neigh(dest).begin();
    auto srcTotal = edges.out_degree(src);
    auto destTotal = edges.out_degree(dest);

    return intersectSortedNodeSetCombined(iter_src, iter_dest, srcTotal, destTotal, 1000, 0.1);

}

static size_t binarySearchIntersectionNeighbor(Graph &edges, NodeID src, NodeID dest) {
    auto iter_src = edges.out_neigh(src).begin();
    auto iter_dest = edges.out_neigh(dest).begin();
    auto srcTotal = edges.out_degree(src);
    auto destTotal = edges.out_degree(dest);

    return intersectSortedNodeSetBinarySearch(iter_src, iter_dest, srcTotal, destTotal);

}


template <typename T>
static int builtin_getVertices(julienne::graph<T> &edges) {
    return edges.n;
}

static VertexSubset<int>* serialSweepCut(Graph& graph,  VertexSubset<int> * vertices, double* val_array){
    //create a copy of the vertex array
    VertexSubset<int>* output_vertexset = new VertexSubset<int>(vertices);

    //sort the vertex array based on the val_array
    output_vertexset->toSparse();
    auto dense_vertex_set = vertices->dense_vertex_set_;
    sort(dense_vertex_set, dense_vertex_set + vertices->num_vertices_,
         [&val_array](const int & a, const int & b) -> bool
         {
             return val_array[a] > val_array[b];
         });

    //find the maximum conductance partitioning

    unordered_set<uintE> S;
    long volS = 0;
    long edgesCrossing = 0;

    double best_conductance = DBL_MAX;
    int best_cut = -1;
    long best_vol = -1;
    long best_edge_cross = -1;

    for (int i = 0; i < vertices->num_vertices_; i++){
        NodeID v = dense_vertex_set[i];
        S.insert(v);
        volS += graph.out_degree(v);
        long denom = (volS < graph.num_edges()-volS)? volS : graph.num_edges()-volS;

        for (NodeID ngh : graph.out_neigh(v)){
            if(S.find(ngh) != S.end()) edgesCrossing--;
            else edgesCrossing++;
        }

        double conductance = (edgesCrossing == 0 || denom == 0) ? 1 : (double)edgesCrossing/denom;

        if(conductance < best_conductance) {
            best_conductance = conductance;
            best_cut = i;
            best_edge_cross = edgesCrossing;
            best_vol = volS;
        }

    }


    //reset the size of the vertex array to the best cut, remove the boolean values
    output_vertexset->num_vertices_ = best_cut;
    output_vertexset->bool_map_ = nullptr;

    return output_vertexset;
}

static int getRandomOutNgh(Graph &edges, NodeID v){
    return edges.get_random_out_neigh(v);
}

static int getRandomInNgh(Graph &edges, NodeID v){
    return edges.get_random_in_neigh(v);
}

static int* serialMinimumSpanningTree(WGraph &edges, NodeID start){
    return minimum_spanning_tree(edges, start);
}

static int * builtin_getOutDegrees(Graph &edges){
    int * out_degrees  = new int [edges.num_nodes()];
    for (NodeID n=0; n < edges.num_nodes(); n++){
        out_degrees[n] = edges.out_degree(n);
    }
    return out_degrees;
}

static uintE * builtin_getOutDegreesUint(Graph &edges){
    uintE * out_degrees  = new uintE [edges.num_nodes()];
    for (NodeID n=0; n < edges.num_nodes(); n++){
        out_degrees[n] = edges.out_degree(n);
    }
    return out_degrees;
}

template <typename T>
static int * builtin_getOutDegrees(julienne::graph<T> &edges) {
    int * out_degrees = new int [edges.n];
    for (uintE n = 0; n < edges.n; n++) {
	    out_degrees[n] = edges.V[n].degree;
    }
    return out_degrees;
}

template <typename T>
static uintE * builtin_getOutDegreesUint(julienne::graph<T> &edges) {
    uintE * out_degrees = new uintE [edges.n];
    for (uintE n = 0; n < edges.n; n++) {
        out_degrees[n] = edges.V[n].degree;
    }
    return out_degrees;
}

static pvector<int> builtin_getOutDegreesPvec(Graph &edges){
    pvector<int> out_degrees (edges.num_nodes(), 0);
    for (NodeID n=0; n < edges.num_nodes(); n++){
        out_degrees[n] = edges.out_degree(n);
    }
    return out_degrees;
}

static int builtin_getVertexSetSize(VertexSubset<int>* vertex_subset){
    return vertex_subset->size();
}

static int builtin_getVertexSetSize(julienne::vertexSubset vs) {
    return vs.size();
}

static void builtin_addVertex(VertexSubset<int>* vertexset, int vertex_id){
    vertexset->addVertex(vertex_id);
}


template <typename T> static void builtin_append (std::vector<T>* vec, T element){
    vec->push_back(element);
}

template <typename T> T static builtin_pop (std::vector<T>* vec){
    T last_element = vec->back();
    vec->pop_back();
    return last_element;
}

//float getTime(){
//    using namespace std::chrono;
//    auto t = high_resolution_clock::now();
//    time_point<high_resolution_clock,microseconds> usec = time_point_cast<microseconds>(t);
//    return (float)(usec.time_since_epoch().count())/1000;
//}

static struct timeval start_time_;
static struct timeval elapsed_time_;

static void startTimer(){
    gettimeofday(&start_time_, NULL);
}

static float stopTimer(){
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
    return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6;

}


static char* argv_safe(int index, char** argv, int argc ){
    // if index is less than or equal to argc than return argv[index]
    //else return false or break command

    if (index < argc) {
        return argv[index];
    } else {
        std::cout << "Error: Did not provide argv[" << index << "] as part of the command line input" << std::endl;
        throw std::invalid_argument( "Did not provide argument" );
    }

}

static Graph builtin_transpose(Graph &graph){
    // Changing this to use shared pointer instead
    //return CSRGraph<NodeID>(graph.num_nodes(), graph.get_in_index_(), graph.get_in_neighbors_(), graph.get_out_index_(), graph.get_out_neighbors_(), true);
      return CSRGraph<NodeID>(graph.num_nodes(), graph.in_index_shared_, graph.in_neighbors_shared_, graph.out_index_shared_, graph.out_neighbors_shared_, true);
}


template<typename APPLY_FUNC> static void builtin_vertexset_apply(VertexSubset<int>* vertex_subset, APPLY_FUNC apply_func){
   if (vertex_subset->is_dense){
       ligra::parallel_for_lambda((int)0, (int)vertex_subset->vertices_range_, [&] (int v) {
               if(vertex_subset->bool_map_[v]){
                   apply_func(v);
               }
           });
   } else {
       if(vertex_subset->dense_vertex_set_ == nullptr && vertex_subset->tmp.size() > 0) {
            ligra::parallel_for_lambda((int)0, (int)vertex_subset->num_vertices_, [&] (int i){
               apply_func(vertex_subset->tmp[i]);
           });
       }else  {
           ligra::parallel_for_lambda((int)0, (int)vertex_subset->num_vertices_, [&] (int i){
               apply_func(vertex_subset->dense_vertex_set_[i]);
           });
       }
   }
}

template<typename OBJECT_TYPE>
static void deleteObject(OBJECT_TYPE* object) {
   if(object)
       delete object;
}
template <typename T>
static VertexSubset<int> * builtin_const_vertexset_filter(T func, int total_elements) {
    VertexSubset<int> * output = new VertexSubset<NodeID>( total_elements, 0);
    bool * next0 = newA(bool, total_elements);
    parallel_for(int v = 0; v < total_elements; v++) {
        next0[v] = 0;
        if (func(v))
            next0[v] = 1;
    }
    output->num_vertices_ = sequence::sum(next0, total_elements);
    output->bool_map_ = next0;
    output->is_dense = true;
    return output;
}

typedef julienne::graph<julienne::symmetricVertex> julienne_graph_type;

static inline julienne_graph_type builtin_loadJulienneEdgesFromFile(std::string filename) {
    char * fname = (char*) filename.c_str();
    return julienne::readGraph<julienne::symmetricVertex>(fname, false, true, false, false);
}

template <typename T>
static inline double to_double(T t) {
    return (double)t;
}

static void deleteObject(julienne::vertexSubset set) {
    set.del();
}

template <typename T>
static VertexSubset<int> * builtin_vertexset_filter(VertexSubset<int> * input, T func) {
    int total_elements = input->vertices_range_;
    //std::cout << "Filter range = " << total_elements << std::endl;
    VertexSubset<int> * output = new VertexSubset<NodeID>( total_elements, 0);
    bool * next0 = newA(bool, total_elements);
    parallel_for(int v = 0; v < total_elements; v++)
        next0[v] = 0;
    if (input->is_dense) {
        //std::cout << "Vertex subset is dense" << std::endl;
        parallel_for(int v = 0; v < total_elements; v++) {
            if (input->bool_map_[v] && func(v))
                next0[v] = 1;
	}
    } else {
        //std::cout << "Vertex subset is sparse" << std::endl;
        if(!(input->dense_vertex_set_ == nullptr && input->num_vertices_ > 0))
            parallel_for(int v = 0; v < input->num_vertices_; v++) {
                //std::cout << "Vertex subset iteration for dense vertex set" << std::endl;
                if (func(input->dense_vertex_set_[v]))
                    next0[input->dense_vertex_set_[v]] = 1;
            }
	else 
            parallel_for(int v = 0; v < input->num_vertices_; v++) {
                //std::cout << "Vertex subset iteration for tmp" << std::endl;
                if (func(input->tmp[v]))
                    next0[input->tmp[v]] = 1;
            }
    }
    output->num_vertices_ = sequence::sum(next0, total_elements);
    output->bool_map_ = next0;
    output->is_dense = true;
    return output;
}

template <typename PriorityType>
  VertexSubset<NodeID> * getBucketWithGraphItVertexSubset(julienne::PriorityQueue<PriorityType>* pq){
    julienne::vertexSubset ready_set = pq->dequeue_ready_set();

    auto vset =  new VertexSubset<NodeID> (ready_set);
//    for (int i = 0; i < vset->num_vertices_; i++){
//        std::cout << "vset[i] vertex: " << vset->dense_vertex_set_[i] << std::endl;
//    }
    return vset;
}


template <typename PriorityType>
void updateBucketWithGraphItVertexSubset(VertexSubset<NodeID>* vset, julienne::PriorityQueue<PriorityType>* pq, bool nodes_init_in_bucket, int delta = 1){
    vset->toSparse();

    if (vset->size() == 0){
        return;
    }

    // Do not insert into overflow bucket since all nodes are in the bucket initially
    if (nodes_init_in_bucket){
        auto f = [&](size_t i) -> julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>> {
            const julienne::uintE v = vset->dense_vertex_set_[i];
            PriorityType null_bkt = pq->get_null_bkt();
            PriorityType priority = (pq->tracking_variable[v] == null_bkt) ? null_bkt : pq->tracking_variable[v]/delta;
//        std::cout << "node: " << v << " priority: " << priority << " tracking val[v]: " << pq->tracking_variable[v] << " bucket: " << pq->get_bucket(priority) << std::endl;
            const julienne::uintE bkt = pq->get_bucket_no_overflow_insertion(priority);
            return julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>>(std::make_tuple(v, bkt));
        };

//    for (int i = 0; i < 5; i++){
//        std::cout << "f[i] vertex: " << std::get<0>(f(i).t) << std::endl;
//        std::cout << "f[i] bkt ID: " << std::get<1>(f(i).t) << std::endl;
//    }

        pq->update_buckets(f, vset->num_vertices_);
    } else {
        auto f = [&](size_t i) -> julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>> {
            const julienne::uintE v = vset->dense_vertex_set_[i];
            PriorityType null_bkt = pq->get_null_bkt();
            PriorityType priority = (pq->tracking_variable[v] == null_bkt) ? null_bkt : pq->tracking_variable[v]/delta;
//        std::cout << "node: " << v << " priority: " << priority << " tracking val[v]: " << pq->tracking_variable[v] << " bucket: " << pq->get_bucket(priority) << std::endl;
            const julienne::uintE bkt = pq->get_bucket_with_overflow_insertion(priority);
            return julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>>(std::make_tuple(v, bkt));
        };

//    for (int i = 0; i < 5; i++){
//        std::cout << "f[i] vertex: " << std::get<0>(f(i).t) << std::endl;
//        std::cout << "f[i] bkt ID: " << std::get<1>(f(i).t) << std::endl;
//    }

        pq->update_buckets(f, vset->num_vertices_);
    }

}


#endif //GRAPHIT_INTRINSICS_H_H
