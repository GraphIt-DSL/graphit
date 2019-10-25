#ifndef GPU_PRIORITY_QUEUE_H
#define GPU_PRIORITY_QUEUE_H

#include <algorithm>
#include <cinttypes>
#include "vertex_frontier.h" 

#ifndef NUM_BLOCKS
  #define NUM_BLOCKS 80
#endif

namespace gpu_runtime {

    template<typename PriorityT_>
    class GPUPriorityQueue;

    static void __global__ update_nodes_identify_min(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices);
  
  
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

    
    gpu_runtime::VertexFrontier __host__ dequeueReadySet(GPUPriorityQueue<PriorityT_> * device_gpq){
      update_nodes_identify_min<<<NUM_BLOCKS, CTA_SIZE>>>(device_gpq, frontier_.max_num_elems);
      return;
    }
    
    PriorityT_* host_priorities_ = nullptr;
    PriorityT_* device_priorities_ = nullptr;
    
    PriorityT_ delta_ = 1;
    PriorityT_ current_priority_ = 0;
    PriorityT_ window_upper_ = 0;

    //Need to do = {0} to avoid dynamic initialization error
    VertexFrontier frontier_ = {0};
    
  };


  static void __global__ update_nodes_identify_min(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices)
  {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int total_work = num_vertices;
    int work_per_thread = (total_work + num_threads - 1)/num_threads;
    int32_t my_minimum = INT_MAX;
    for (int i = 0; i < work_per_thread; i++) {
      int32_t node_id = thread_id + i * num_threads;
	if (node_id < num_vertices) {
	  if (gpq->device_priorities_[node_id] >= (gpq->window_upper_) && gpq->device_priorities_[node_id] != INT_MAX && gpq->device_priorities_[node_id] < my_minimum) {
	    my_minimum = gpq->device_priorities_[node_id];
	  }
	}
    }
    
    if (my_minimum < gpq->current_priority_){
          atomicMin(&(gpq->current_priority_), my_minimum);
    }
  }

  
}


#endif // GPU_PRIORITY_QUEUE_H
