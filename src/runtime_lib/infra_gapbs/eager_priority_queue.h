#include <algorithm>
#include <cinttypes>

#include "platform_atomics.h"


/**
 * Phase-synchronous priority queue with dual representation
 * Representation 1: When using thread-local buckets, there is nothing stored in the data strucutre. It merely holds the current bucket index, next bucket index and other metadata. The real priority queue is distributed across threads. 
 * Representation 2: When using lazy buckets, the priority queue actually stores all the nodes with their buckets (the buckets are not distributed)
 **/
template<typename PriorityT_>
class EagerPriorityQueue {

public:
  explicit EagerPriorityQueue(PriorityT_* priorities) {
    priorities_ = priorities;
    init_indexes_tails();
  }

  // set up shared_indexes, iter and frontier_tails data structures
  void init_indexes_tails(){
    shared_indexes[0] = 0;
    shared_indexes[1] = kMaxBin;
    frontier_tails[0] = 1;
    frontier_tails[1] = 0;

    iter_=0;
  }

  // get the prioirty of the current iteration (each iter has a priority)
  size_t get_current_priority(){
    return shared_indexes[iter_&1];
  }

  // increment the iteration number, which was used for computing the current priorty
  void increment_iter() {
    iter_++;
  }
  
  bool finished() {
      return get_current_priority() == kMaxBin;
  }

  void updatePriorityMin(NodeID dst, PriorityT_ new_p, PriorityT_ old_p){

  }

  PriorityT_* priorities_;
  const PriorityT_ kDistInf = numeric_limits<PriorityT_>::max()/2;
  const size_t kMaxBin = numeric_limits<size_t>::max()/2;

  size_t shared_indexes[2];
  size_t frontier_tails[2];
  size_t iter_;;

};
