#pragma once
#include <algorithm>
#include <cinttypes>

//#include "platform_atomics.h"
#include "bucket.h"


typedef int64_t NodeID;



// The functor is not required for the new API
/*
template<class Priority>
class PriorityFunctor {

 public:
  PriorityFunctor(){};

 PriorityFunctor(Priority* priority): priority_(priority){};

  Priority virtual operator()(NodeID v){
    return priority_[v];
  }

  Priority* priority_;
  
};
*/

template<class D>
class PriorityQueue {

public:
  D* tracking_variable;
  explicit PriorityQueue(size_t n, D* priority_array, bucket_order bkt_order, priority_order pri_order, size_t total_buckets=128) {
    
    //cout << "constructing a priority map from array" << endl;
    buckets_ = new buckets<D*>(n, priority_array, bkt_order, pri_order, total_buckets);
    tracking_variable = priority_array;
    cur_priority_ = 0;
  }


  // get the prioirty of the current iteration (each iter has a priority)
  D get_current_priority(){
    return cur_priority_;
  }

  inline bucket next_bucket() {
    auto next_bkt =  buckets_->next_bucket();
    cur_priority_ = next_bkt.id;
    return next_bkt;
  }

  template <class F>
  inline size_t update_buckets(F f, size_t k) {
    return buckets_->update_buckets(f, k);
  }

  inline bucket_dest get_bucket(const bucket_id& next) const {
    return buckets_->get_bucket(next);
  }

  ~PriorityQueue(){
    //cout << "destructing Priority Queue " << endl;
    delete buckets_;
  }
  
  buckets<D*>* buckets_;
  uintE cur_priority_ = 0;
  
  inline bool finished(void) {
      return cur_priority_ == buckets_->null_bkt;
  }
  inline julienne::vertexSubset dequeue_ready_set() {
	auto bucket = next_bucket();
	return bucket.identifiers;
  }
  void updatePrioritySum(uintE a, int b, D c) {
  
  }
  void updatePrioritySum(uintE a, int b) {
  }
  D* get_tracking_variable(void) {
    return tracking_variable;
  }
  void deleteObject(void) {
    delete this;
  } 
};
