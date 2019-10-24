#ifndef GPU_PRIORITY_QUEUE_H
#define GPU_PRIORITY_QUEUE_H

#include <algorithm>
#include <cinttypes>
#include "vertex_frontier.h" 

namespace gpu_runtime {
  
  template<typename PriorityT_>
    class GPUPriorityQueue {
    
  public:

    size_t get_current_priority(){
      return current_priority_;
    }
    
    void update_current_priority(PriorityT_ priority_change_){
      
    }
    
    bool finished() {
      //TODO
      return true;
    }
    
    bool host_finishedNode(NodeID v){
      return host_priorities_[v]/delta_ < get_current_priority();;
    }

    bool __device__ device_finishedNode(NodeID v){

    }
    
    PriorityT_* host_priorities_ = nullptr;
    PriorityT_* device_priorities_ = nullptr;
    
    PriorityT_ delta_ = 1;
    PriorityT_ current_priority_ = 0;
    PriorityT_ window_upper_ = 0;
    
  };
}


#endif // GPU_PRIORITY_QUEUE_H
