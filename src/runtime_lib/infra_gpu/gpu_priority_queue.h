#ifndef GPU_PRIORITY_QUEUE_H
#define GPU_PRIORITY_QUEUE_H

#include <algorithm>
#include <cinttypes>
#include "vertex_frontier.h" 

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 80
#endif

#ifndef CTA_SIZE
#define CTA_SIZE 1024
#endif


namespace gpu_runtime {

template<typename PriorityT_>
	class GPUPriorityQueue;

static void __global__ update_nodes_identify_min(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices);
static void __global__ update_nodes_special(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices, gpu_runtime::VertexFrontier output_frontier);
static void __device__ update_nodes_identify_min_device(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices);
static void __device__ update_nodes_special_device(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices, gpu_runtime::VertexFrontier output_frontier);

template<typename PriorityT_>
	class GPUPriorityQueue {

	public:

		size_t getCurrentPriority(){
			return current_priority_;
		}

		void init(GraphT<int32_t> graph, PriorityT_ * host_priorities, PriorityT_* device_priorities, PriorityT_ initial_priority, PriorityT_ delta, NodeID initial_node = -1){
			host_priorities_ = host_priorities;
			device_priorities_ = device_priorities;
			current_priority_ = initial_priority;
			delta_ = delta;
			ready_set_dequeued = false;
			frontier_ = gpu_runtime::create_new_vertex_set(gpu_runtime::builtin_getVertices(graph));
			frontier_.d_priority_array = device_priorities;
			frontier_.priority_cutoff = current_priority_ + delta_;
			cudaMalloc(&current_priority_shared, sizeof(PriorityT_));
			if (initial_node != -1){
				gpu_runtime::builtin_addVertex(frontier_, initial_node);
			}
		}
		void release(void) {
			delete_vertex_frontier(frontier_);
		}

		void __device__ updatePriorityMin(GPUPriorityQueue<PriorityT_> * device_gpq,  PriorityT_ new_priority, VertexFrontier output_frontier, int32_t node){
			bool output = gpu_runtime::writeMin(&(device_gpq->device_priorities_[node]), new_priority);
			if (device_gpq->device_priorities_[node] >= (device_gpq->current_priority_ + device_gpq->delta_)) return;
			if (output){
				enqueueVertexBytemap(output_frontier.d_byte_map_output, output_frontier.d_num_elems_output, node);
			}

		}

		bool finished(GPUPriorityQueue<PriorityT_> * device_gpq) {
			if (current_priority_ == INT_MAX){
				return true;
			}

			if (!ready_set_dequeued && gpu_runtime::builtin_getVertexSetSize(frontier_) == 0){
				dequeueReadySet(device_gpq);
				ready_set_dequeued = true;
				return current_priority_ == INT_MAX;
			} 

			return false;
		}
#ifdef GLOBAL
		bool __device__ device_finished(void) {
			if (current_priority_ == INT_MAX)
				return true;
			if (!ready_set_dequeued && gpu_runtime::device_builtin_getVertexSetSize(frontier_) == 0) {
				device_dequeueReadySet();
				if (threadIdx.x + blockIdx.x * blockDim.x == 0)
					ready_set_dequeued = true;
				this_grid().sync();
				return current_priority_ == INT_MAX;
			}
			return false;
		}
#endif
		bool __device__ device_finished(void) {
			if(current_priority_ == INT_MAX)
				return true;
			if (!ready_set_dequeued && gpu_runtime::device_builtin_getVertexSetSize(frontier_) == 0) {
				device_dequeueReadySet();
				ready_set_dequeued = true;
				return current_priority_ == INT_MAX;
			}
			return false;
		}

		bool host_finishedNode(NodeID v){
			return host_priorities_[v]/delta_ < current_priority_;
		}

		bool __device__ device_finishedNode(NodeID v){

		}

		VertexFrontier& dequeueReadySet(GPUPriorityQueue<PriorityT_> * device_gpq){
			// if this is already dequeued in the previous finish() operator
			// then don't do the dequeu operation again
			if (ready_set_dequeued){
				//Now that we dequeued it, the next ready set is no longer dequeued
				ready_set_dequeued = false;
				return frontier_;
			}

			//perform the dequeue operation only if the current frontier is empty
			if (gpu_runtime::builtin_getVertexSetSize(frontier_) == 0) {
				window_upper_ = current_priority_ + delta_;
				current_priority_ = INT_MAX;

				cudaMemcpy(current_priority_shared, &current_priority_, sizeof(int32_t), cudaMemcpyHostToDevice);
				cudaMemcpy(device_gpq, this, sizeof(*device_gpq), cudaMemcpyHostToDevice); 
				gpu_runtime::cudaCheckLastError();
				update_nodes_identify_min<<<NUM_BLOCKS, CTA_SIZE>>>(device_gpq, frontier_.max_num_elems);
				gpu_runtime::cudaCheckLastError();

				cudaMemcpy(&(device_gpq->current_priority_), current_priority_shared, sizeof(int32_t), cudaMemcpyDeviceToHost);

				cudaMemcpy(this, device_gpq, sizeof(*this), cudaMemcpyDeviceToHost);
				gpu_runtime::cudaCheckLastError();

				update_nodes_special<<<NUM_BLOCKS, CTA_SIZE>>>(device_gpq, frontier_.max_num_elems,  frontier_);
				gpu_runtime::cudaCheckLastError();
				gpu_runtime::swap_queues(frontier_);
				frontier_.format_ready = gpu_runtime::VertexFrontier::SPARSE;

				//Now that we dequeued it, the next ready set is no longer dequeued
				frontier_.priority_cutoff = current_priority_ + delta_;
				ready_set_dequeued = false;
				return frontier_;
			}

			//if it is empty, just return the empty frontier
			return frontier_;
		}
		
		VertexFrontier __device__ device_dequeueReadySet(void) {
			if (ready_set_dequeued) {
				ready_set_dequeued = false;
				return frontier_;
			}
			if (gpu_runtime::device_builtin_getVertexSetSize(frontier_) == 0) {
				window_upper_ = current_priority_ + delta_;
				current_priority_ = INT_MAX;
				this_grid().sync();
				if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
					current_priority_shared[0] = INT_MAX;
				}
				this_grid().sync();
				
				update_nodes_identify_min_device(this, frontier_.max_num_elems);
				this_grid().sync();
				
				current_priority_ = current_priority_shared[0];
				this_grid().sync();
				update_nodes_special_device(this, frontier_.max_num_elems, frontier_);
				gpu_runtime::swap_queues_device(frontier_);
				frontier_.format_ready = gpu_runtime::VertexFrontier::SPARSE;
				ready_set_dequeued = false;
				frontier_.priority_cutoff = current_priority_ + delta_;
				return frontier_;
			}
			return frontier_;
		}	


#ifdef GLOBAL
		VertexFrontier __device__ device_dequeueReadySet(void) {
/*
			if (threadIdx.x + blockDim.x * blockIdx.x == 0)
				printf("Entering dequeue ready set\n");
*/
			if (ready_set_dequeued) {
				this_grid().sync();
				if (threadIdx.x + blockIdx.x * blockDim.x == 0)
					ready_set_dequeued = false;
				this_grid().sync();
				return frontier_;
			}
			if (gpu_runtime::device_builtin_getVertexSetSize(frontier_) == 0) {
/*				
				if (threadIdx.x + blockDim.x * blockIdx.x == 0)
					printf("Entering special case\n");
*/
				this_grid().sync();
				if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
					window_upper_ = current_priority_ + delta_;
					current_priority_ = INT_MAX;
				}
				this_grid().sync();
				// No need for copy
				update_nodes_identify_min_device(this, frontier_.max_num_elems);	
				this_grid().sync();
				update_nodes_special_device(this, frontier_.max_num_elems, frontier_);
				this_grid().sync();
				gpu_runtime::swap_queues_device_global(frontier_);
				this_grid().sync();	
				if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
					frontier_.format_ready = gpu_runtime::VertexFrontier::SPARSE;
					ready_set_dequeued = false;
				}
				this_grid().sync();
				return frontier_;
					
			}
			this_grid().sync();
			return frontier_;
		}
#endif

		PriorityT_* host_priorities_ = nullptr;
		PriorityT_* device_priorities_ = nullptr;

		PriorityT_ delta_ = 1;
		PriorityT_ current_priority_ = 0;
		PriorityT_ window_upper_ = 0;

		//Need to do = {0} to avoid dynamic initialization error
		VertexFrontier frontier_ = {0};
		bool ready_set_dequeued = false;
		
		PriorityT_ *current_priority_shared = nullptr;
	};


static void __device__ update_nodes_identify_min_device(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices) {
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

	if (my_minimum < gpq->current_priority_shared[0]){
		atomicMin(&(gpq->current_priority_shared[0]), my_minimum);
	}
}//end of update_nodes_identify_min



static void __device__ update_nodes_special_device(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices, gpu_runtime::VertexFrontier output_frontier){

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	//int warp_id = thread_id / 32;

	int total_work = num_vertices;
	int work_per_thread = (total_work + num_threads - 1)/num_threads;
	for (int i = 0; i < work_per_thread; i++) {
		int32_t node_id = thread_id + i * num_threads;
		if (node_id < num_vertices) {
			if(gpq->device_priorities_[node_id] >= gpq->current_priority_ && gpq->device_priorities_[node_id] < (gpq->current_priority_ + gpq->delta_)) {
				gpu_runtime::enqueueVertexSparseQueue(output_frontier.d_sparse_queue_output, output_frontier.d_num_elems_output, node_id);
			}
		}
	}
}


static void __global__ update_nodes_identify_min(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices) {
	update_nodes_identify_min_device(gpq, num_vertices);	
}

static void __global__ update_nodes_special(GPUPriorityQueue<int32_t>* gpq,  int32_t num_vertices, gpu_runtime::VertexFrontier output_frontier){
	update_nodes_special_device(gpq, num_vertices, output_frontier);
}


}


#endif // GPU_PRIORITY_QUEUE_H
