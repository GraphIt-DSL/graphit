#include "gpu_intrinsics.h"
#include <algorithm>


#define USE_DEDUP 0
#define SORT_NODES 0
#include <assert.h>
#include <vector>
#include <queue>

//#define DEBUG

#ifdef DEBUG
  #define ITER_COUNT (5)
#else
  #define ITER_COUNT (1)
#endif

gpu_runtime::GPUPriorityQueue<int> host_gpq;
gpu_runtime::GPUPriorityQueue<int> __device__  device_gpq; 

typedef struct {
	int32_t *SP;
	int32_t *output_size;
	int32_t num_blocks;
	int32_t *node_borders;
	int32_t *edge_borders;
	int32_t *old_indices;
	int32_t window_lower;
	int32_t window_upper;		
	int32_t *new_window_start;
}algo_state;

int32_t __device__ *SP;
int32_t *__host_SP;
int32_t *__device_SP;

//int32_t __device__ window_lower;
//int32_t __device__ window_upper;


#define VIRTUAL_WARP_SIZE (32)
#define NUM_THREADS (1024)
#define NUM_BLOCKS (80)
#define CTA_SIZE (1024)
#define WARP_SIZE (32)
#define STAGE_1_SIZE (8)

void __global__ init_kernel(gpu_runtime::GraphT<int32_t> graph, algo_state device_state, int start_v) {
        int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int total_work = graph.num_vertices;
        int work_per_thread = (total_work + num_threads - 1)/num_threads;
        for (int i = 0; i < work_per_thread; i++) {
                int id = num_threads * i + thread_id;
                if (id < total_work) {
					device_state.SP[id] = INT_MAX;
                }
        }
	if (thread_id == 0) {
		//reset with the new data structure
		SP[start_v] = 0;
		device_state.SP[start_v] = 0;
	}
}

bool __device__ updateEdge(int32_t src, int32_t dst, int32_t weight) {
	bool output2;
	bool SP_trackving_var_1 = 0;
	SP_trackving_var_1 = gpu_runtime::writeMin(&SP[dst], (SP[src] + weight));
	output2 = SP_trackving_var_1;

	//if (SP[dst] >= window_upper) return false;
	if (SP[dst] >= (device_gpq.current_priority_ + device_gpq.delta_)) return false;
	
	return output2;
}

template <typename EdgeWeightType>
void __device__ gpu_operator_body_3(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	// Body of the actual operator code
	EdgeWeightType weight = graph.d_edge_weight[edge_id];
	if (updateEdge(src, dst, weight)){
		//gpu_runtime::enqueueVertexSparseQueue(output_frontier.d_sparse_queue_output, output_frontier.d_num_elems_output, dst);
		gpu_runtime::enqueueVertexBytemap(output_frontier.d_byte_map_output, output_frontier.d_num_elems_output, dst);
	}
}

void __global__ update_nodes_identify_min(gpu_runtime::GraphT<int32_t> graph, algo_state device_state) {
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	
	int total_work = graph.num_vertices;
	int work_per_thread = (total_work + num_threads - 1)/num_threads;
	int32_t my_minimum = INT_MAX;
	for (int i = 0; i < work_per_thread; i++) {
		int32_t node_id = thread_id + i * num_threads;
		if (node_id < graph.num_vertices) {
		  if (SP[node_id] >= (device_gpq.window_upper_) && SP[node_id] != INT_MAX && SP[node_id] < my_minimum) {
				my_minimum = SP[node_id];
			}
		}
	}
	//if (my_minimum < device_state.new_window_start[0]) {
	if (my_minimum < device_gpq.current_priority_){
	  //atomicMin(device_state.new_window_start, my_minimum);
	  atomicMin(&(device_gpq.current_priority_), my_minimum);
	}	
}
void __global__ update_nodes_special(gpu_runtime::GraphT<int32_t> graph, algo_state device_state,  gpu_runtime::VertexFrontier output_frontier) {
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	//int warp_id = thread_id / 32;	
	
	int total_work = graph.num_vertices;
	int work_per_thread = (total_work + num_threads - 1)/num_threads;
	for (int i = 0; i < work_per_thread; i++) {
		int32_t node_id = thread_id + i * num_threads;
		if (node_id < graph.num_vertices) {
		  //if(SP[node_id] >= device_state.window_lower && SP[node_id] < device_state.window_upper) {
		  if(SP[node_id] >= device_gpq.current_priority_ && SP[node_id] < (device_gpq.current_priority_ + device_gpq.delta_)) {
				gpu_runtime::enqueueVertexSparseQueue(output_frontier.d_sparse_queue_output, output_frontier.d_num_elems_output, node_id);
			}	
		}
	}
}
void allocate_state(algo_state &host_state, algo_state &device_state, gpu_runtime::GraphT<int32_t> &graph) {
	host_state.SP = new int[graph.num_vertices];
	host_state.output_size = new int32_t[1];
	host_state.new_window_start = new int32_t[1];
	cudaMalloc(&device_state.SP, sizeof(int32_t)*graph.num_vertices);	
	cudaMalloc(&device_state.output_size, sizeof(int32_t));
	cudaMalloc(&device_state.new_window_start, sizeof(int32_t));
}

void swap_pointers(int32_t **a, int32_t **b) {
	int32_t* t = *a;
	*a = *b;
	*b = t;
}

void __device__ SP_generated_vector_op_apply_func_0(int32_t v) {
	SP[v] = 2147483647;
}


int main(int argc, char *argv[]) {
	cudaSetDevice(0);
	cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
	gpu_runtime::GraphT<int32_t> graph;
	gpu_runtime::load_graph(graph, argv[1], false);
	int32_t delta = atoi(argv[3]);
	int32_t start_vertex = atoi(argv[2]);
	
	cudaMalloc(&__device_SP, gpu_runtime::builtin_getVertices(graph) * sizeof(int32_t));
	cudaMemcpyToSymbol(SP, &__device_SP, sizeof(int32_t*), 0);
	
	__host_SP = new int32_t[gpu_runtime::builtin_getVertices(graph)];

	algo_state host_state, device_state;	
	allocate_state(host_state, device_state, graph);

	cudaDeviceSynchronize();
	
	float total_time = 0;
	for (int outer = 0; outer < ITER_COUNT; outer++) {
		float iter_total = 0;
		//this sets it to Sparse
		gpu_runtime::VertexFrontier frontier = gpu_runtime::create_new_vertex_set(gpu_runtime::builtin_getVertices(graph));
		gpu_runtime::builtin_addVertex(frontier, start_vertex);
		gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorAll, SP_generated_vector_op_apply_func_0><<<NUM_CTA, CTA_SIZE>>>(graph.getFullFrontier());
		startTimer();

		host_gpq.delta_ = delta;
		host_gpq.current_priority_ = 0 ;

		cudaMemcpyToSymbol(device_gpq, &host_gpq, sizeof(host_gpq), 0);
		gpu_runtime::cudaCheckLastError();
		
		init_kernel<<<NUM_BLOCKS, CTA_SIZE>>>(graph, device_state, start_vertex);
		gpu_runtime::cudaCheckLastError();
		
		int iters = 0;	
		cudaDeviceSynchronize();
		float t = stopTimer();
		//printf("Init time = %f\n", t);
		iter_total+=t;

		//while(gpu_runtime::builtin_getVertexSetSize(frontier) != (0)){
		while(! host_gpq.finished()){
			startTimer();
			iters++;
			gpu_runtime::vertex_set_prepare_sparse(frontier);
			//cudaMemcpyToSymbol(window_upper, &device_state.window_upper, sizeof(int32_t), 0);
			//Might not be necessary, always synchronized at this point?? 
			cudaMemcpyToSymbol(device_gpq, &host_gpq, sizeof(host_gpq), 0);
			gpu_runtime::cudaCheckLastError();

			//gpu_runtime::vertex_based_load_balance_host<int32_t, gpu_operator_body_3, gpu_runtime::AccessorSparse, gpu_runtime::true_function>(graph, frontier, frontier);  
			gpu_runtime::TWCE_load_balance_host<int32_t, gpu_operator_body_3, gpu_runtime::AccessorSparse, gpu_runtime::true_function>(graph, frontier, frontier);
			gpu_runtime::cudaCheckLastError();
			
			gpu_runtime::swap_bytemaps(frontier);
			// set the input to the prepare function
			frontier.format_ready = gpu_runtime::VertexFrontier::BYTEMAP;
			
			if (gpu_runtime::builtin_getVertexSetSize(frontier) == (0)) {
			  //host_state.new_window_start[0] = INT_MAX;
			  host_gpq.window_upper_ = host_gpq.current_priority_ + host_gpq.delta_;
			  host_gpq.current_priority_ = INT_MAX;
			  
			  cudaMemcpyToSymbol(device_gpq, &host_gpq, sizeof(host_gpq), 0);
			  gpu_runtime::cudaCheckLastError();

			  update_nodes_identify_min<<<NUM_BLOCKS, CTA_SIZE>>>(graph, device_state);
			  gpu_runtime::cudaCheckLastError();
			  cudaMemcpyFromSymbol(&host_gpq, device_gpq, sizeof(host_gpq), 0,cudaMemcpyDeviceToHost);
			  gpu_runtime::cudaCheckLastError();

			  //if(host_gpq.current_priority_ == INT_MAX){
			  //  break;
			  //}			  
			  update_nodes_special<<<NUM_BLOCKS, CTA_SIZE>>>( graph, device_state, frontier);
			  gpu_runtime::cudaCheckLastError();
			  gpu_runtime::swap_queues(frontier);
			  frontier.format_ready = gpu_runtime::VertexFrontier::SPARSE; 
			}

			cudaDeviceSynchronize();
			t = stopTimer();

			#ifdef DEBUG
			//printf("Iter %d output_size = %d \n", iters, gpu_runtime::builtin_getVertexSetSize(frontier));
			#endif
			
			iter_total += t;
		}


		#ifdef DEBUG
		printf("Num iters = %d\n", iters);
		printf("Time elapsed = %f\n", iter_total);
		#endif
		
		total_time += iter_total;

	}

	#ifdef DEBUG
	printf("Total time = %f\n", total_time);
	#endif
	
	if (argc > 3)
		if (argv[4][0] == 'v'){ 
			//FILE *output = fopen("output.txt", "w");
			cudaMemcpy(host_state.SP, __device_SP, sizeof(int32_t)*graph.num_vertices, cudaMemcpyDeviceToHost);
			#ifdef DEBUG
			FILE *output = fopen("output.txt", "w");
			#endif
			
			for (int i = 0; i < graph.num_vertices; i++){
				#ifdef DEBUG
				fprintf(output, "%d, %d\n", i, host_state.SP[i]);
				#else
				printf("%d\n", host_state.SP[i]);
                                #endif
			}
		}
	return 0;
}
