#include "gpu_intrinsics.h"
#include <algorithm>

#define ITER_COUNT (1)
#define USE_DEDUP 0
#define SORT_NODES 0
#include <assert.h>
#include <vector>
#include <queue>


typedef struct {
	int32_t *SP;

	int32_t *frontier1;
	int32_t *frontier2;

	int32_t *frontier1_size;
	int32_t *frontier2_size;
	int32_t *iters;

	int32_t *output_size;

	int32_t num_blocks;

	int32_t *node_borders;
	int32_t *edge_borders;

	int32_t *worklist;
	int32_t *old_indices;
}algo_state;

//struct timeval start_time_;
//struct timeval elapsed_time_;

// void startTimer(){
// 	gettimeofday(&start_time_, NULL);
// }

// float stopTimer(){
// 	gettimeofday(&elapsed_time_, NULL);
// 	elapsed_time_.tv_sec  -= start_time_.tv_sec;
// 	elapsed_time_.tv_usec -= start_time_.tv_usec;
// 	return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6;
// }

void cudaCheckLastError(void) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
}


#define VIRTUAL_WARP_SIZE (32)
#define NUM_THREADS (1024)
#define NUM_BLOCKS (80)
#define CTA_SIZE (1024)
#define WARP_SIZE (32)
#define STAGE_1_SIZE (8)

void __global__ init_kernel(gpu_runtime::GraphT<int32_t> graph, algo_state device_state, int32_t start_vertex) {
        int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int total_work = graph.num_vertices;
        int work_per_thread = (total_work + num_threads - 1)/num_threads;
        for (int i = 0; i < work_per_thread; i++) {
                int id = num_threads * i + thread_id;
                if (id < total_work) {
			device_state.SP[id] = INT_MAX;
			device_state.iters[id] = 0;
                }
        }
	if (thread_id == 0) {
		device_state.SP[start_vertex] = 0;
		//starting point is set to 0 
		device_state.frontier1[0] = start_vertex;
		*device_state.frontier1_size = 1;
		*device_state.frontier2_size = 0;
	}
}
__device__ inline int warp_bcast(int v, int leader) { return __shfl_sync(__activemask(), v, leader); }
__device__ inline int atomicAggInc(int *ctr) {
	int32_t lane_id = threadIdx.x % 32;
	
        int mask = __activemask();
        int leader = __ffs(mask) - 1;
        int res;
        if(lane_id == leader)
                res = atomicAdd(ctr, __popc(mask));
        res = warp_bcast(res, leader);

        return (res + __popc(mask & ((1 << lane_id) - 1)));
}
__device__ void enqueueVertex(int32_t v, algo_state &device_state, int32_t curr_iter) {
	if (device_state.iters[v] == curr_iter)
		return;
	device_state.iters[v] = curr_iter;
	int32_t pos = atomicAggInc(device_state.frontier2_size);
	device_state.frontier2[pos] = v;
}

void __global__ update_edges (gpu_runtime::GraphT<int32_t> graph, algo_state device_state, int32_t curr_iter) {
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	
	int lane_id = thread_id % 32;

	__shared__ int32_t stage2_queue[CTA_SIZE];
	__shared__ int32_t stage3_queue[CTA_SIZE];
	__shared__ int32_t stage_queue_sizes[3];
	if (threadIdx.x == 0) {
		stage_queue_sizes[0] = 0;
		stage_queue_sizes[1] = 0;
		stage_queue_sizes[2] = 0;
	}
	__syncthreads();
	
	
	__shared__ int32_t stage2_offset[CTA_SIZE];
	__shared__ int32_t stage3_offset[CTA_SIZE];

	__shared__ int32_t stage2_size[CTA_SIZE];
	__shared__ int32_t stage3_size[CTA_SIZE];
	

	int32_t total_vertices = device_state.frontier1_size[0];	

	int32_t my_vertex_idx = thread_id / (STAGE_1_SIZE);
	int32_t d;
	int32_t s1_offset;
	int32_t my_vertex;
	int32_t row_offset;
	if (my_vertex_idx < total_vertices) {
		my_vertex = device_state.frontier1[my_vertex_idx];
		// Step 1 segreggate vertices into shared buffers	
		if (thread_id % (STAGE_1_SIZE) == 0 ) {
			d = graph.d_get_degree(my_vertex);
			row_offset = graph.d_src_offsets[my_vertex];	
			int32_t s3_size = d/CTA_SIZE;
			d = d - s3_size * CTA_SIZE;
			if (s3_size) {
				int32_t pos = atomicAggInc(&stage_queue_sizes[2]);
				stage3_queue[pos] = my_vertex;			
				stage3_size[pos] = s3_size * CTA_SIZE;
				// stage3_offset[pos] = 0; // Not required because always 0
				stage3_offset[pos] = row_offset;	
			}
			
			int32_t s2_size = d/WARP_SIZE;
			d = d - s2_size * WARP_SIZE;
			
			if (s2_size) {
				int32_t pos = atomicAggInc(&stage_queue_sizes[1]);
				stage2_queue[pos] = my_vertex;
				stage2_offset[pos] = s3_size * CTA_SIZE + row_offset;
				stage2_size[pos] = s2_size * WARP_SIZE;
			}
			s1_offset = s3_size * CTA_SIZE + s2_size * WARP_SIZE + row_offset;
		}
	}else
		my_vertex = -1;

	__syncthreads();
	
	d = __shfl_sync((uint32_t)-1, d, (lane_id / STAGE_1_SIZE) * STAGE_1_SIZE, 32);
	s1_offset = __shfl_sync((uint32_t)-1, s1_offset, (lane_id / STAGE_1_SIZE) * STAGE_1_SIZE, 32);
	int32_t src_distance;
	if (my_vertex_idx < total_vertices) {
		// STAGE 1	
		my_vertex = device_state.frontier1[my_vertex_idx];
		src_distance = device_state.SP[my_vertex];
		for (int32_t neigh_id = s1_offset + (lane_id % STAGE_1_SIZE); neigh_id < d + s1_offset; neigh_id += STAGE_1_SIZE) {
			// DO ACTUAL SSSP
			int32_t dst = graph.d_edge_dst[neigh_id];
			int32_t new_dst = graph.d_edge_weight[neigh_id] + src_distance;
			if (new_dst < device_state.SP[dst]) {
				atomicMin(&device_state.SP[dst], new_dst);
				enqueueVertex(dst, device_state, curr_iter);
			}	
		}		
	}	
	// STAGE 2 -- stage 2 is dynamically balanced
	while (1) {
		int32_t to_process;
		if (lane_id == 0) {
			to_process = atomicSub(&stage_queue_sizes[1], 1) - 1;	
		}
		to_process = __shfl_sync((uint32_t)-1, to_process, 0, 32);
		if (to_process < 0)
			break;
		my_vertex = stage2_queue[to_process];
		d = stage2_size[to_process];
		int32_t s2_offset = stage2_offset[to_process];	
		src_distance = device_state.SP[my_vertex];
		
		for (int32_t neigh_id = s2_offset + (lane_id); neigh_id < d + s2_offset; neigh_id += WARP_SIZE) {
			// DO ACTUAL SSSP
			int dst = graph.d_edge_dst[neigh_id];
			int new_dst = graph.d_edge_weight[neigh_id] + src_distance;
			if (new_dst < device_state.SP[dst]) {
				atomicMin(&device_state.SP[dst], new_dst);
				enqueueVertex(dst, device_state, curr_iter);
			}	
		}
	}	

	// STAGE 3 -- all threads have to do all, no need for LB
	for (int32_t wid = 0; wid < stage_queue_sizes[2]; wid ++) {
		my_vertex = stage3_queue[wid];
		d = stage3_size[wid];
		int32_t s3_offset = stage3_offset[wid];
		src_distance = device_state.SP[my_vertex];
		
		for (int32_t neigh_id = s3_offset + (threadIdx.x); neigh_id < d + s3_offset; neigh_id += CTA_SIZE) {
			// DO ACTUAL SSSP
			int dst = graph.d_edge_dst[neigh_id];
			int new_dst = graph.d_edge_weight[neigh_id] + src_distance;
			if (new_dst < device_state.SP[dst]) {
				atomicMin(&device_state.SP[dst], new_dst);
				enqueueVertex(dst, device_state, curr_iter);
			}	
		}
	}	
}
void __global__ update_nodes (gpu_runtime::GraphT<int32_t> graph, algo_state device_state) {
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	
	int total_work = graph.num_vertices;
	int work_per_thread = (total_work + num_threads - 1)/num_threads;
	
	for (int i = 0; i < work_per_thread; i++) {
		int32_t node_id = thread_id + i * num_threads;
		if (node_id < graph.num_vertices) {
			if (device_state.frontier2[node_id]) {
				device_state.frontier2[node_id] = 0;
				int pos = atomicAdd(device_state.frontier1_size, 1);
				device_state.frontier1[pos] = node_id;
			}
		}
	}

}
void allocate_state(algo_state &host_state, algo_state &device_state, gpu_runtime::GraphT<int32_t> &graph) {
	host_state.SP = new int[graph.num_vertices];
	host_state.output_size = new int32_t[1];

	host_state.frontier1_size = new int32_t[1];
	host_state.frontier1 = new int32_t[graph.num_vertices];

	
	cudaMalloc(&device_state.SP, sizeof(int32_t)*graph.num_vertices);	

	cudaMalloc(&device_state.frontier1, sizeof(int32_t)*graph.num_vertices * 6);	
	cudaMalloc(&device_state.frontier2, sizeof(int32_t)*graph.num_vertices * 6);	
	cudaMalloc(&device_state.iters, sizeof(int32_t)*graph.num_vertices);	

	cudaMalloc(&device_state.frontier1_size, sizeof(int32_t));	
	cudaMalloc(&device_state.frontier2_size, sizeof(int32_t));	

	cudaMalloc(&device_state.output_size, sizeof(int32_t));


	cudaMalloc(&device_state.worklist, sizeof(int32_t));
}

void swap_pointers(int32_t **a, int32_t **b) {
	int32_t* t = *a;
	*a = *b;
	*b = t;
}
void swap_queues(algo_state &device_state) {
	swap_pointers(&device_state.frontier1, &device_state.frontier2);
	swap_pointers(&device_state.frontier1_size, &device_state.frontier2_size);
}


template <typename EdgeWeightType>
void __device__ gpu_operator_body_3(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	// Body of the actual operator code
	EdgeWeightType weight = graph.d_edge_weight[edge_id];
	//if (updateEdge(src, dst, weight)) {
		gpu_runtime::enqueueVertexSparseQueue(output_frontier.d_sparse_queue_output, output_frontier.d_num_elems_output, dst);
		//}
}


int main(int argc, char *argv[]) {
	cudaSetDevice(0);
	cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
	gpu_runtime::GraphT<int32_t> graph;
	gpu_runtime::load_graph(graph, argv[1], false);
	int32_t start_vertex = atoi(argv[2]);

	algo_state host_state, device_state;

	allocate_state(host_state, device_state, graph);

	cudaDeviceSynchronize();
		
	float total_time = 0;
	for (int outer = 0; outer < ITER_COUNT; outer++) {
		float iter_total = 0;
		startTimer();
		
		startTimer();
		init_kernel<<<NUM_BLOCKS, CTA_SIZE>>>(graph, device_state, start_vertex);		
		int iters = 0;	
		cudaDeviceSynchronize();
		float t = stopTimer();
		//printf("Init time = %f\n", t);
		iter_total+=t;

		host_state.frontier1_size[0] = 1;
		while(*host_state.frontier1_size) {
			startTimer();
			iters++;
			int num_threads = *host_state.frontier1_size *(STAGE_1_SIZE);
			int num_cta = (num_threads + CTA_SIZE-1)/CTA_SIZE;
			
			update_edges<<<num_cta, CTA_SIZE>>>(graph, device_state, iters);
			//gpu_runtime::vertex_based_load_balance_host<int32_t, gpu_operator_body_3, gpu_runtime::AccessorSparse, gpu_runtime::true_function>(edges, frontier, frontier);
			
			host_state.frontier1_size[0] = 0;
			cudaMemcpy(device_state.frontier1_size, host_state.frontier1_size, sizeof(int32_t), cudaMemcpyHostToDevice);
			
			swap_queues(device_state);
				
			cudaCheckLastError();
			cudaMemcpy(host_state.frontier1_size, device_state.frontier1_size, sizeof(int32_t), cudaMemcpyDeviceToHost);

			t = stopTimer();
			//printf("Iter %d time = %f, output_size = %d <%d, %d>\n", iters, t, *host_state.frontier1_size, num_cta, CTA_SIZE);
			iter_total += t;
		}
		
		//printf("Num iters = %d\n", iters);
		//printf("Time elapsed = %f\n", iter_total);
		total_time += iter_total;

	}
	//printf("Total time = %f\n", total_time);
	if (argc > 3)
		if (argv[3][0] == 'v'){ 
			//FILE *output = fopen("output.txt", "w");
			cudaMemcpy(host_state.SP, device_state.SP, sizeof(int32_t)*graph.num_vertices, cudaMemcpyDeviceToHost);
			for (int i = 0; i < graph.num_vertices; i++)
				//fprintf(output, "%d, %d\n", i, host_state.SP[i]);
				printf("%d\n", host_state.SP[i]);
		}else if (argv[2][0] == 'c'){
			/*
			for (int i = 0; i < NUM_BLOCKS * NUM_THREADS; i++)
				printf("%d: %d\n", i, counters[i]);
			*/
		}

	return 0;

}
