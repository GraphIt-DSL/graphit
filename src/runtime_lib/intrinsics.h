//
// Created by Yunming Zhang on 4/25/17.
//

#ifndef GRAPHIT_INTRINSICS_H_H
#define GRAPHIT_INTRINSICS_H_H


#include <vector>

#include "infra_gapbs/builder.h"
#include "infra_gapbs/benchmark.h"
#include "infra_gapbs/bitmap.h"
#include "infra_gapbs/command_line.h"
#include "infra_gapbs/graph.h"
#include "infra_gapbs/platform_atomics.h"
#include "infra_gapbs/pvector.h"
#include <queue>
#include <curses.h>
#include "infra_gapbs/timer.h"
#include "infra_gapbs/sliding_queue.h"
#include "edgeset_apply_functions.h"

#include "infra_ligra/ligra/ligra.h"

#include "vertexsubset.h"

#include <time.h>
#include <chrono>



template <typename T>
T builtin_sum(T* input_vector, int num_elem){
    //Serial Code for summation
    //T output_sum = 0;
    //for (int i = 0; i < num_elem; i++){
    //    output_sum += input_vector[i];
    //}

    T reduce_sum = sequence::plusReduce(input_vector, num_elem);

    return reduce_sum;
}

//For now, assume the weights are ints, this would be good enough for now
// Later, we can change the parser, to supply type information to the library call
WGraph builtin_loadWeightedEdgesFromFile(std::string file_name){
    CLBase cli (file_name);
    WeightedBuilder weighted_builder (cli);
    WGraph g = weighted_builder.MakeGraph();
    return g;
}

Graph builtin_loadEdgesFromFile(std::string file_name){
    CLBase cli (file_name);
    Builder builder (cli);
    Graph g = builder.MakeGraph();
    return g;
}

int builtin_getVertices(Graph &edges){
    return edges.num_nodes();
}

int builtin_getVertices(WGraph &edges){
    return edges.num_nodes();
}

int * builtin_getOutDegrees(Graph &edges){
    int * out_degrees  = new int [edges.num_nodes()];
    for (NodeID n=0; n < edges.num_nodes(); n++){
        out_degrees[n] = edges.out_degree(n);
    }
    return out_degrees;
}

pvector<int> builtin_getOutDegreesPvec(Graph &edges){
    pvector<int> out_degrees (edges.num_nodes(), 0);
    for (NodeID n=0; n < edges.num_nodes(); n++){
        out_degrees[n] = edges.out_degree(n);
    }
    return out_degrees;
}

int builtin_getVertexSetSize(VertexSubset<int>* vertex_subset){
    return vertex_subset->size();
}

void builtin_addVertex(VertexSubset<int>* vertexset, int vertex_id){
    vertexset->addVertex(vertex_id);
}

template <typename T> void builtin_append (std::vector<T>* vec, T element){
    vec->push_back(element);
}

template <typename T> T builtin_pop (std::vector<T>* vec){
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

struct timeval start_time_;
struct timeval elapsed_time_;

void startTimer(){
    gettimeofday(&start_time_, NULL);
}

float stopTimer(){
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
    return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6;

}

Graph builtin_transpose(Graph &graph){
    return CSRGraph<NodeID>(graph.num_nodes(), graph.in_index_, graph.in_neighbors_, graph.out_index_, graph.out_neighbors_, true);
}


template<typename APPLY_FUNC> void builtin_vertexset_apply(VertexSubset<int>* vertex_subset, APPLY_FUNC apply_func){
   if (vertex_subset->is_dense){
       parallel_for (int v = 0; v < vertex_subset->vertices_range_; v++){
           if(vertex_subset->bool_map_[v]){
               apply_func(v);
           }
       }
   } else {
       parallel_for (int i = 0; i < vertex_subset->num_vertices_; i++){
           apply_func(vertex_subset->dense_vertex_set_[i]);
       }
   }
}

template<typename OBJECT_TYPE>
void deleteObject(OBJECT_TYPE* object) {
   if(object)
       delete object;
}

#endif //GRAPHIT_INTRINSICS_H_H
