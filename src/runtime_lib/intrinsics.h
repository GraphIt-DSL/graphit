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
#include "infra_gapbs/minimum_spanning_tree.h"


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
	EdgeList el;
	for (NodeID x = 0; x < num_nodes; x++)
		for(int32_t _y = indptr[x]; _y < indptr[x+1]; _y++)
			el.push_back(Edge(x, indices[_y]));
	CLBase cli(0, NULL);
	BuilderBase<NodeID> bb(cli);
	return bb.MakeGraphFromEL(el);
}
static WGraph builtin_loadWeightedEdgesFromCSR(const int32_t *data, const int32_t *indptr, const NodeID *indices, int num_nodes, int num_edges) {
	typedef EdgePair<NodeID, WNode> Edge;
	typedef pvector<Edge> EdgeList;
	EdgeList el;
	for (NodeID x = 0; x < num_nodes; x++) {
		for (int32_t _y = indptr[x]; _y < indptr[x+1]; _y++) {
			el.push_back(Edge(x, NodeWeight<NodeID, WeightT>(indices[_y], data[_y])));
		}
	}	
	CLBase cli(0, NULL);
	BuilderBase<NodeID, WNode, WeightT> bb(cli);
	bb.needs_weights_ = false;
	return bb.MakeGraphFromEL(el);	
}
static int builtin_getVertices(Graph &edges){
    return edges.num_nodes();
}

static int builtin_getVertices(WGraph &edges){
    return edges.num_nodes();
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
static void deleteObject(OBJECT_TYPE* object) {
   if(object)
       delete object;
}

#endif //GRAPHIT_INTRINSICS_H_H
