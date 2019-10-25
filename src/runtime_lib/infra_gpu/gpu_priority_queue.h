#ifndef GPU_PRIORITY_QUEUE_H
#define GPU_PRIORITY_QUEUE_H

#include <algorithm>
#include <cinttypes>
#include "vertex_frontier.h" 


namespace gpu_runtime {
  
  template<typename PriorityT_>
    class GPUPriorityQueue {
    
  public:

    size_t getCurrentPriority(){
      return current_priority_;
    }

    void init(PriorityT_ * host_priorities, PriorityT_* device_priorities, PriorityT_ initial_priority, PriorityT_ delta, NodeID initial_node = -1){
      host_priorities_ = host_priorities;
      device_priorities_ = device_priorities;
      current_priority_ = initial_priority;
      delta_ = delta;
      if (initial_node != -1){
	//if (frontier_ != {0}){
	  gpu_runtime::builtin_addVertex(frontier_, initial_node);
	  //}
      }
    }
    
    void updatePriorityMin(PriorityT_ priority_change_){
      
    }
    
    bool finished() {
      return current_priority_ == INT_MAX;
    }
    
    bool host_finishedNode(NodeID v){
      return host_priorities_[v]/delta_ < current_priority_;
    }

    bool __device__ device_finishedNode(NodeID v){

    }

    

    gpu_runtime::VertexFrontier __device__ dequeueReadySet(){
      

    }
    
    PriorityT_* host_priorities_ = nullptr;
    PriorityT_* device_priorities_ = nullptr;
    
    PriorityT_ delta_ = 1;
    PriorityT_ current_priority_ = 0;
    PriorityT_ window_upper_ = 0;

    //Need to do = {0} to avoid dynamic initialization error
    VertexFrontier frontier_ = {0};

    
  };
}


#endif // GPU_PRIORITY_QUEUE_H
