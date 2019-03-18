#include <algorithm>
#include <cinttypes>

//#include "platform_atomics.h"
#include "bucket.h"


typedef int64_t NodeID;

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


template<class D>
class PriorityQueue {

public:
  explicit PriorityQueue(size_t n, bucket_order bkt_order, D* d,  priority_order pri_order, size_t total_buckets=128) {
    buckets_ = new buckets<D>(n, d, bkt_order, pri_order, total_buckets);
    cur_priority_ = 0;
  }

  template <class P>
  explicit PriorityQueue(size_t n, P* priority_array, bucket_order bkt_order, priority_order pri_order, size_t total_buckets=128) {
    
    //cout << "constructing a priority map from array" << endl;
    PriorityFunctor<P>* pfw = new PriorityFunctor<P>(priority_array);
    buckets_ = new buckets<PriorityFunctor<P>>(n, pfw, bkt_order, pri_order, total_buckets);

    cur_priority_ = 0;
  }


  // get the prioirty of the current iteration (each iter has a priority)
  uintE get_current_priority(){
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
  
  buckets<D>* buckets_;
  uintE cur_priority_ = 0;
  
  inline bool finished(void) {
      return cur_priority_ == buckets_->null_bkt;
  }

};
