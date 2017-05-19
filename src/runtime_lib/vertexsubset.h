//
// Created by Yunming Zhang on 5/18/17.
//

#ifndef GRAPHIT_VERTEXSUBSET_H
#define GRAPHIT_VERTEXSUBSET_H

#include <cinttypes>
#include <iostream>
#include <type_traits>

template <typename NodeID_>
struct VertexSubset {
    int64_t vertices_range_, num_vertices_;
    bool* boolean_index_vector_;
    bool is_dense;

    // make a singleton vertex in range of n
//    VertexSubset(int64_t vertices_range, NodeID_ v)
//            : vertices_range_(vertices_range), num_vertices_(1), index_vector_(NULL), is_dense(0) {
//        index_vector_[v] = true;
//    }

    //empty vertex set
    VertexSubset(int64_t vertices_range) : num_vertices_(0), vertices_range_(vertices_range), is_dense(0) {
        boolean_index_vector_ = new bool(vertices_range);
    }

    // make vertexSubset from array of vertex indices
    // n is range, and m is size of array

    // make vertexSubset from boolean array, where n is range

    // make vertexSubset from boolean array giving number of true values

    // delete the contents
    void del(){
        if (boolean_index_vector_ != NULL) free(boolean_index_vector_);
    }

    bool contains(NodeID_ v){
        return boolean_index_vector_[v];
    }

    bool addVertex(NodeID_ v){
        boolean_index_vector_[v] = true;
    }

    long getVerticesRange() { return vertices_range_; }
    long size() { return num_vertices_; }
    bool isEmpty() { return num_vertices_==0; }

    // converts to dense but keeps sparse representation if there
    void toDense() {
//        if (d == NULL) {
//            d = newA(bool,n);
//            {parallel_for(long i=0;i<n;i++) d[i] = 0;}
//            {parallel_for(long i=0;i<m;i++) d[s[i]] = 1;}
//        }
        is_dense = true;
    }

    // converts to sparse but keeps dense representation if there
    void toSparse() {

    }

};

#endif //GRAPHIT_VERTEXSUBSET_H
