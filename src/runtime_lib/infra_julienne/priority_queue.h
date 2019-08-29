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
  D null_bkt = std::numeric_limits<D>::max();
  int delta_;


  explicit PriorityQueue(size_t n, D* priority_array, bucket_order bkt_order, priority_order pri_order, size_t total_buckets=128, int delta = 1) {

      //cout << "constructing a priority map from array" << endl;
      buckets_ = new buckets<D>(n, priority_array, bkt_order, pri_order, total_buckets, delta);
      tracking_variable = priority_array;
      cur_priority_ = 0;
      delta_ = delta;
  }



  // get the prioirty of the current iteration (each iter has a priority)
  D get_current_priority(){
    return cur_priority_;
  }

  D get_null_bkt(){
    return null_bkt;

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

  // Do not return the overflow bucket as a possible bucket for insertion (because all nodes are inserted in overflow initially)
  inline bucket_dest get_bucket_no_overflow_insertion(const bucket_id& next) const {
    return buckets_->get_bucket_no_overflow_insertion(next);
  }

  // Allow returning overflow bucket as a possible bucket for insertion (nodes are not initially inserted in the bucket)
  inline bucket_dest get_bucket_with_overflow_insertion(const bucket_id& next) const {
      return buckets_->get_bucket_with_overflow_insertion(next);
  }

  ~PriorityQueue(){
    //cout << "destructing Priority Queue " << endl;
    delete buckets_;
  }
  
  buckets<D>* buckets_;
  uintE cur_priority_ = 0;
  
  inline bool finished(void) {
      return cur_priority_ == buckets_->null_bkt;
  }

  inline bool finishedNode(uintE node){
      return cur_priority_ >= tracking_variable[node]/delta_;
  }

  inline julienne::vertexSubset dequeue_ready_set() {
	auto bucket = next_bucket();
	return bucket.identifiers;
  }

  // perform the update if the current value is greater than minimum_val
  // the final value should also be no less than minimum_val
  bool updatePrioritySum(uintE v, D delta, D minimum_val) {
      if (tracking_variable[v] <= minimum_val)
          return false;
      D old_val = tracking_variable[v];
      tracking_variable[v]= tracking_variable[v] + (delta);
      if (tracking_variable[v] < minimum_val)
          tracking_variable[v] = minimum_val;
      return true;
  }

  bool updatePrioritySumAtomic(uintE v, D delta, D minimum_val){
      if (tracking_variable[v] <= minimum_val )
          return false;
      writeAdd(&tracking_variable[v], delta);
      if (tracking_variable[v] < minimum_val)
          CAS(&tracking_variable[v], tracking_variable[v], minimum_val);
      return true;
  }

  bool updatePriorityMinAtomic(uintE v, D old_val, D new_val){
      return writeMin(&tracking_variable[v], new_val);
  }

  bool updatePriorityMin(uintE v, D old_val, D new_val){
      if (tracking_variable[v] <= new_val){
          return false;
      } else {
          tracking_variable[v] = new_val;
          return true;
      }
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
