//
// Created by Yunming Zhang on 5/18/17.
//

#ifndef GRAPHIT_VERTEXSUBSET_H
#define GRAPHIT_VERTEXSUBSET_H

#include <cinttypes>
#include <iostream>
#include <type_traits>
#include "infra_gapbs/sliding_queue.h"
#include "infra_ligra/ligra/parallel.h"
#include "infra_ligra/ligra/utils.h"


template <typename NodeID_>
struct VertexSubset {
    int64_t vertices_range_, num_vertices_;
    bool is_dense;
    //SlidingQueue<NodeID>* dense_vertex_set_;
    unsigned int* dense_vertex_set_;
    Bitmap * bitmap_ ;
    std::vector<NodeID> tmp;
    bool* bool_map_;
    SlidingQueue<NodeID>* sliding_queue_;


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

        if (num_vertices == vertices_range){

            //try not to initialize unncessary data structures, this can be expensive for PageRank, which returns full set
            bitmap_ = new Bitmap(vertices_range);
            bitmap_->set_all();
            bool_map_ = newA(bool, vertices_range);
            parallel_for(int i = 0; i < vertices_range; i++) bool_map_[i] = 1;

            dense_vertex_set_ = new unsigned int[vertices_range];
// don't need this for now
//            sliding_queue_ = new SlidingQueue<NodeID>(vertices_range);
            parallel_for (NodeID i = 0; i< vertices_range; i++){
                dense_vertex_set_[i] = i;
                //hopefully we will only need to use one of the two in the futuer (dense_set or sliding queue)
                //sliding_queue_->push_back(i);
            }
            sliding_queue_ = nullptr;

        } else {
            bool_map_ = nullptr;
            bitmap_ = nullptr;
            dense_vertex_set_ = nullptr;
            sliding_queue_ = nullptr;
        }
    }

    // make vertexSubset from array of vertex indices
    // n is range, and m is size of array

    // make vertexSubset from boolean array, where n is range

    // make vertexSubset from boolean array giving number of true values

    // delete the contents
     ~VertexSubset(){
	if(dense_vertex_set_)
		delete[] dense_vertex_set_;
	if(bitmap_)
		delete bitmap_;
	if(bool_map_)
		delete[] bool_map_;
    }

    bool contains(NodeID_ v){
        return bitmap_->get_bit(v);
    }

    void addVertex(NodeID_ v){
        //only increment the count if the vertex is not already in the vertexset
        if (bitmap_ == nullptr){
            bitmap_ = new Bitmap(vertices_range_);
            bitmap_->reset();
        }

        if (bool_map_ == nullptr){
            bool_map_ = newA(bool, vertices_range_);
            parallel_for(int i = 0; i < vertices_range_; i++) bool_map_[i] = 0;
        }

        if (sliding_queue_ == nullptr){
            // initialized to two times of vertices range
            // (one for the frontier to be read, the other for the frontier to be written to)
            sliding_queue_ = new SlidingQueue<NodeID>(2*vertices_range_);
        }
        // TODO: this is a hack for now, need to solve it later. Sliding window needs to be called before usage
        sliding_queue_->push_back(v);
        //sliding_queue_->slide_window();

        if (!bitmap_->get_bit(v)){
            bitmap_->set_bit(v);
            //dense_vertex_set_->push_back(v);
            bool_map_[v] = 1;
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

        if (bool_map_ == NULL) {
            bool_map_ = newA(bool,vertices_range_);
            {parallel_for(long i=0;i<vertices_range_;i++) bool_map_[i] = 0;}

            if (tmp.size() != 0){
                for (NodeID node : tmp){
                    //bitmap_->set_bit(node);
                    bool_map_[node] = 1;
                }
            } else if (num_vertices_ > 0){
                {parallel_for(long i=0;i<num_vertices_;i++) bool_map_[dense_vertex_set_[i]] = 1;}
            }
        }

//        if (bitmap_ == nullptr){
//            //set only if bitvector is not yet created
//            //and temporary vector is created
//            if (tmp.size() != 0){
//                for (NodeID node : tmp){
//                    bitmap_->set_bit(node);
//                }
//            } else if (num_vertices_ > 0){
//                parallel_for (int i = 0; i < num_vertices_; i++){
//                    bitmap_->set_bit_atomic(dense_vertex_set_[i]);
//                }
//            }
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
        if (dense_vertex_set_ == nullptr && tmp.size() > 0) {
            dense_vertex_set_ = new unsigned int[num_vertices_];
            for (int i = 0; i < num_vertices_; i++) {
                dense_vertex_set_[i] = tmp[i];
            }

        }else if (dense_vertex_set_ == nullptr && num_vertices_ > 0){

                _seq<uintE> R = sequence::packIndex<uintE>(bool_map_,vertices_range_);
                if (num_vertices_ != R.n) {
		  cout << "num_vertices_: " << num_vertices_ << " R.n: " << R.n << endl;
                    cout << "bad stored value of m" << endl;
                    abort();
                }
                dense_vertex_set_ = R.A;

        }

//        } else if (dense_vertex_set_ == nullptr && num_vertices_ > 0){
//            //vertices are stored as bitvector
//            dense_vertex_set_ = new unsigned int[num_vertices_];
//            int j = 0;
//            for (unsigned int i = 0; i < vertices_range_; i++){
//                if (bitmap_->get_bit(i)){
//                    dense_vertex_set_[j] = i;
//                    j++;
//                }
//            }
//        }

    }

};

#endif //GRAPHIT_VERTEXSUBSET_H
