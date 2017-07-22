//
// Created by Yunming Zhang on 5/18/17.
//

#ifndef GRAPHIT_VERTEXSUBSET_H
#define GRAPHIT_VERTEXSUBSET_H

#include <cinttypes>
#include <iostream>
#include <type_traits>
#include "infra_gapbs/sliding_queue.h"


template <typename NodeID_>
struct VertexSubset {
    int64_t vertices_range_, num_vertices_;
    bool is_dense;
    //SlidingQueue<NodeID>* dense_vertex_set_;
    unsigned int* dense_vertex_set_;
    Bitmap * bitmap_ ;
    std::vector<NodeID> tmp;

    // make a singleton vertex in range of n
//    VertexSubset(int64_t vertices_range, NodeID_ v)
//            : vertices_range_(vertices_range), num_vertices_(1), index_vector_(NULL), is_dense(0) {
//        index_vector_[v] = true;
//    }

    //set every vertex to true in the vertex subset
    VertexSubset(int64_t vertices_range, int64_t num_vertices)
            : num_vertices_(num_vertices),
              vertices_range_(vertices_range),
              is_dense(0)
    {
        bitmap_ = new Bitmap(vertices_range);
        bitmap_->reset();
        if (num_vertices == vertices_range){
            bitmap_->set_all();
            dense_vertex_set_ = new unsigned int[vertices_range];
            for (int i = 0; i< vertices_range; i++){
                dense_vertex_set_[i] = i;
            }
        } else {
            dense_vertex_set_ = nullptr;
        }
    }

    // make vertexSubset from array of vertex indices
    // n is range, and m is size of array

    // make vertexSubset from boolean array, where n is range

    // make vertexSubset from boolean array giving number of true values

    // delete the contents
     ~VertexSubset(){
    }

    bool contains(NodeID_ v){
        return bitmap_->get_bit(v);
    }

    void addVertex(NodeID_ v){
        //only increment the count if the vertex is not already in the vertexset
        if (!bitmap_->get_bit(v)){
            bitmap_->set_bit(v);
            //dense_vertex_set_->push_back(v);
            num_vertices_++;
            tmp.push_back(v);
        }
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

    void printDenseSet(){
        bool first = true;
        std::cout << "dense set: ";
        for (int i = 0; i < num_vertices_; i++){
            if (first){
                std::cout << dense_vertex_set_[i];
                first = false;
            } else {
                std::cout << ", " << dense_vertex_set_[i];
            }
        }
        std::cout << std::endl;
    }

    // converts to sparse but keeps dense representation if there
    void toSparse() {
        if (dense_vertex_set_ == nullptr && tmp.size() > 0){
            dense_vertex_set_ = new unsigned int[num_vertices_];
            for (int i = 0; i < num_vertices_; i++){
                dense_vertex_set_[i] = tmp[i];
            }
        }
    }

};

#endif //GRAPHIT_VERTEXSUBSET_H
