#ifndef GPU_PRIORITY_QUEUE_H
#define GPU_PRIORITY_QUEUE_H

#include <algorithm>
#include <cinttypes>
#include "vertex_frontier.h" 

namespace gpu_runtime {

  template<typename PriorityT_>
    class GPUPriorityQueue {
    
  public:
    explicit GPUPriorityQueue(PriorityT_* priorities, PriorityT_ delta=1)
      : priorities_(priorities), delta_(delta){
    }
    
    size_t get_current_priority(){
      return current_priority_;
    }

	void update_current_priority(PriorityT_ priority_change_){

	}
    
    bool finished() {
      //TODO
      return true;
    }
    
    bool finishedNode(NodeID v){
		return priorities_[v]/delta_ < get_current_priority();;
    }
    
    PriorityT_* priorities_;
    PriorityT_ delta_;
	PriorityT_ current_priority_;
  };
}


#endif // GPU_PRIORITY_QUEUE_H
