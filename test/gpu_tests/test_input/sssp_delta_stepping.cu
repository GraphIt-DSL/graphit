


#define VIRTUAL_WARP_SIZE (32)
#define NUM_THREADS (1024)
#define NUM_BLOCKS (80)
#define CTA_SIZE (1024)
#define WARP_SIZE (32)
#define STAGE_1_SIZE (8)


#include "gpu_intrinsics.h"
#include <algorithm>


#define USE_DEDUP 0
#define SORT_NODES 0
#include <assert.h>

//#define DEBUG

#ifdef DEBUG
  #define ITER_COUNT (5)
#else
  #define ITER_COUNT (1)
#endif

gpu_runtime::GPUPriorityQueue<int> host_gpq;
gpu_runtime::GPUPriorityQueue<int> __device__  device_gpq; 


int32_t __device__ *SP;
int32_t *__host_SP;
int32_t *__device_SP;


void __global__ init_kernel(gpu_runtime::GraphT<int32_t> graph, int start_v) {
        int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int total_work = graph.num_vertices;
        int work_per_thread = (total_work + num_threads - 1)/num_threads;
	if (thread_id == 0) {
		//reset with the new data structure
		SP[start_v] = 0;
	}
}

/*bool __device__ updateEdge(int32_t src, int32_t dst, int32_t weight) {
        bool output2;
        bool SP_trackving_var_1 = 0;
	SP_trackving_var_1 = gpu_runtime::writeMin(&SP[dst], (SP[src] + weight));
	output2 = SP_trackving_var_1;
	if (SP[dst] >= (device_gpq.current_priority_ + device_gpq.delta_)) return false;
	return output2;
	}*/

void __device__ deviceUpdateEdge(int32_t src, int32_t dst, int32_t weight, gpu_runtime::VertexFrontier output_frontier){
  device_gpq.updatePriorityMin(&device_gpq, (SP[src] + weight), output_frontier, dst);
}

template <typename EdgeWeightType>
void __device__ gpu_operator_body_3(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	// Body of the actual operator code
	EdgeWeightType weight = graph.d_edge_weight[edge_id];
	deviceUpdateEdge(src, dst, weight, output_frontier);
	/*if (updateEdge(src, dst, weight)){
		gpu_runtime::enqueueVertexBytemap(output_frontier.d_byte_map_output, output_frontier.d_num_elems_output, dst);
		}*/
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
	cudaDeviceSynchronize();
	float total_time = 0;
	for (int outer = 0; outer < ITER_COUNT; outer++) {
		float iter_total = 0;
		//this sets it to Sparse
		//host_gpq.frontier_ = gpu_runtime::create_new_vertex_set(gpu_runtime::builtin_getVertices(graph));
		
		gpu_runtime::vertex_set_apply_kernel<gpu_runtime::AccessorAll, SP_generated_vector_op_apply_func_0><<<NUM_CTA, CTA_SIZE>>>(graph.getFullFrontier());
		startTimer();

		host_gpq.init(graph, __host_SP, __device_SP, 0, delta, start_vertex);

		cudaMemcpyToSymbol(device_gpq, &host_gpq, sizeof(host_gpq), 0);
		gpu_runtime::cudaCheckLastError();
		
		init_kernel<<<NUM_BLOCKS, CTA_SIZE>>>(graph, start_vertex);
		gpu_runtime::cudaCheckLastError();
		
		int iters = 0;	
		cudaDeviceSynchronize();
		float t = stopTimer();
		//printf("Init time = %f\n", t);
		iter_total+=t;

		gpu_runtime::GPUPriorityQueue<int> * tmp_gpq;
		cudaGetSymbolAddress(((void **)&tmp_gpq), device_gpq);
		
		while(! host_gpq.finished(tmp_gpq)){
			startTimer();
			iters++;
			
			gpu_runtime::VertexFrontier& frontier = host_gpq.dequeueReadySet(tmp_gpq);
			
			gpu_runtime::vertex_set_prepare_sparse(frontier);
			cudaMemcpyToSymbol(device_gpq, &host_gpq, sizeof(host_gpq), 0);
			gpu_runtime::cudaCheckLastError();

			gpu_runtime::TWCE_load_balance_host<int32_t, gpu_operator_body_3, gpu_runtime::AccessorSparse, gpu_runtime::true_function>(graph, frontier, frontier);
			gpu_runtime::cudaCheckLastError();

			gpu_runtime::swap_bytemaps(frontier);
			// set the input to the prepare function
			frontier.format_ready = gpu_runtime::VertexFrontier::BYTEMAP;
			
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
			cudaMemcpy(__host_SP, __device_SP, sizeof(int32_t)*graph.num_vertices, cudaMemcpyDeviceToHost);
			#ifdef DEBUG
			FILE *output = fopen("output.txt", "w");
			#endif
			
			for (int i = 0; i < graph.num_vertices; i++){
				#ifdef DEBUG
				fprintf(output, "%d, %d\n", i, __host_SP[i]);
				#else
				printf("%d\n", __host_SP[i]);
                                #endif
			}
		}
	return 0;
}
