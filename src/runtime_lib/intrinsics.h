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
#include <unordered_map>
#include <unordered_set>

#include "infra_ligra/ligra/ligra.h"

#include "vertexsubset.h"

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
       if(vertex_subset->dense_vertex_set_ == nullptr && vertex_subset->tmp.size() > 0) {
           parallel_for (int i = 0; i < vertex_subset->num_vertices_; i++){
               apply_func(vertex_subset->tmp[i]);
           }
       }else  {
           parallel_for (int i = 0; i < vertex_subset->num_vertices_; i++){
               apply_func(vertex_subset->dense_vertex_set_[i]);
           }
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
#endif //GRAPHIT_INTRINSICS_H_H
