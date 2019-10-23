#include "gpu_intrinsics.h"
#include <stdio.h>
//#include <cub/cub.cuh>
//#include <cooperative_groups.h>

gpu_runtime::GraphT<int32_t> edges;
int32_t __device__ *SP;
int32_t *__host_SP;
int32_t *__device_SP;
template <typename EdgeWeightType>
void __device__ gpu_operator_body_3(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	// Body of the actual operator code
	EdgeWeightType weight = graph.d_edge_weight[edge_id];
	if (updateEdge(src, dst, weight)) {
		gpu_runtime::enqueueVertexBytemap(output_frontier.d_byte_map_output, output_frontier.d_num_elems_output, dst);
	}
}

template <typename EdgeWeightType>
void __global__ gpu_operator_kernel_1 (gpu_runtime::GraphT<EdgeWeightType> graph, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	gpu_runtime::vertex_based_load_balance<EdgeWeightType, gpu_operator_body_3<EdgeWeightType>, gpu_runtime::AccessorSparse, gpu_runtime::true_function> (graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType>
void __global__ gpu_operator_kernel_2 (gpu_runtime::GraphT<EdgeWeightType> graph, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	gpu_runtime::warp_based_load_balance<EdgeWeightType, gpu_operator_body_3<EdgeWeightType>, gpu_runtime::AccessorSparse, gpu_runtime::true_function> (graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType>
void __global__ gpu_operator_kernel_3 (gpu_runtime::GraphT<EdgeWeightType> graph, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	gpu_runtime::tb_based_load_balance<EdgeWeightType, gpu_operator_body_3<EdgeWeightType>, gpu_runtime::AccessorSparse, gpu_runtime::true_function> (graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType>
void __global__ gpu_operator_kernel_4 (gpu_runtime::GraphT<EdgeWeightType> graph, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	gpu_runtime::STRICT_load_balance<EdgeWeightType, gpu_operator_body_3<EdgeWeightType>, gpu_runtime::AccessorSparse, gpu_runtime::true_function> (graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType>
void __global__ gpu_operator_kernel_5_mid (gpu_runtime::GraphT<EdgeWeightType> graph, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	gpu_runtime::TWC_load_balance_mid<EdgeWeightType, gpu_operator_body_3<EdgeWeightType>, gpu_runtime::AccessorSparse, gpu_runtime::true_function> (graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType>
void __global__ gpu_operator_kernel_5_large (gpu_runtime::GraphT<EdgeWeightType> graph, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	gpu_runtime::TWC_load_balance_large<EdgeWeightType, gpu_operator_body_3<EdgeWeightType>, gpu_runtime::AccessorSparse, gpu_runtime::true_function> (graph, input_frontier, output_frontier);
}

template <typename EdgeWeightType>
void __global__ gpu_operator_kernel_6 (gpu_runtime::GraphT<EdgeWeightType> graph, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {
	gpu_runtime::TWCE_load_balance<EdgeWeightType, gpu_operator_body_3<EdgeWeightType>, gpu_runtime::AccessorSparse, gpu_runtime::true_function> (graph, input_frontier, output_frontier);
}


void __device__ SP_generated_vector_op_apply_func_0(int32_t v) {
	SP[v] = 2147483647;
}
bool __device__ updateEdge(int32_t src, int32_t dst, int32_t weight) {
	bool output2;
	bool SP_trackving_var_1 = 0;
	SP_trackving_var_1 = gpu_runtime::writeMin(&SP[dst], (SP[src] + weight));
	output2 = SP_trackving_var_1;
	return output2;
}
void __device__ reset(int32_t v) {
	SP[v] = 2147483647;
}
int __host__ main(int argc, char* argv[]) {
	gpu_runtime::load_graph(edges, argv[1], false);
	cudaMalloc(&__device_SP, gpu_runtime::builtin_getVertices(edges) * sizeof(int32_t));
	cudaMemcpyToSymbol(SP, &__device_SP, sizeof(int32_t*), 0);
	__host_SP = new int32_t[gpu_runtime::builtin_getVertices(edges)];
	gpu_runtime::vertex_set_apply_kernel<SP_generated_vector_op_apply_func_0><<<NUM_CTA, CTA_SIZE>>>(gpu_runtime::builtin_getVertices(edges));

        int dev;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        int32_t mp = deviceProp.multiProcessorCount*2;

	for (int32_t trail = 0; trail < 1; trail++) {
		gpu_runtime::vertex_set_apply_kernel<reset><<<NUM_CTA, CTA_SIZE>>>(gpu_runtime::builtin_getVertices(edges));
		int32_t n = gpu_runtime::builtin_getVertices(edges);
		gpu_runtime::VertexFrontier frontier = gpu_runtime::create_new_vertex_set(gpu_runtime::builtin_getVertices(edges));


		int32_t start_vertex = atoi(argv[2]);
		gpu_runtime::builtin_addVertex(frontier, start_vertex);
		__host_SP[start_vertex] = 0;
		cudaMemcpy(__device_SP + start_vertex, __host_SP + start_vertex, sizeof(int32_t), cudaMemcpyHostToDevice);
		int32_t rounds = 0;

		startTimer();
		while ((gpu_runtime::builtin_getVertexSetSize(frontier)) != (0)) {
			gpu_runtime::VertexFrontier output = frontier;
			{
				gpu_runtime::vertex_set_prepare_sparse(frontier);
				output = frontier;
				int32_t num_cta, cta_size;

#ifdef TM
				gpu_runtime::vertex_based_load_balance_info<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_operator_kernel_1<<<num_cta, cta_size>>>(edges, frontier, output);
#endif
#ifdef WM
				gpu_runtime::warp_based_load_balance_info<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_operator_kernel_2<<<num_cta, cta_size>>>(edges, frontier, output);
#endif
#ifdef CM
				gpu_runtime::tb_based_load_balance_info<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_operator_kernel_3<<<num_cta, cta_size>>>(edges, frontier, output);
#endif
#ifdef STRICT
				gpu_runtime::STRICT_load_balance_info<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_runtime::STRICT_gather<gpu_runtime::AccessorSparse><<<num_cta, cta_size>>>(edges, frontier);

				int32_t f_size = builtin_getVertexSetSize(frontier);				
				int32_t tot_elt;

#ifdef XXX
//int *ttt=(int *)malloc(sizeof(int)*f_size);
//cudaMemcpy(ttt, &frontier.d_sparse_queue_input[edges.num_vertices], sizeof(int)*f_size, cudaMemcpyDeviceToHost);
//for(int i=0;i<f_size;i++) printf("%d ", ttt[i]); printf("\n");


				tot_elt = gpu_runtime::GPU_prefix_sum(mp, &frontier.d_sparse_queue_input[edges.num_vertices], &frontier.d_sparse_queue_input[edges.num_vertices*2], f_size);
				num_cta = (tot_elt+CTA_SIZE-1)/CTA_SIZE;

//cudaMemcpy(ttt, &frontier.d_sparse_queue_input[edges.num_vertices], sizeof(int)*f_size, cudaMemcpyDeviceToHost);
//for(int i=0;i<f_size;i++) printf("%d ", ttt[i]); printf("\n");
//fprintf(stdout, "tot: %d %d\n", f_size, tot_elt);
#endif

#ifdef YYY
				int32_t *tmp = (int32_t *)malloc(sizeof(int32_t)*(f_size+1));
				cudaMemcpy(&tmp[1], &frontier.d_sparse_queue_input[edges.num_vertices], sizeof(int32_t)*f_size, cudaMemcpyDeviceToHost);
				tmp[0] = 0;
				for(int i=1; i<=f_size;i++) {
					tmp[i] = tmp[i] + tmp[i-1];
				} 

				cudaMemcpy(&frontier.d_sparse_queue_input[edges.num_vertices], tmp, sizeof(int32_t)*(f_size+1), cudaMemcpyHostToDevice);
				num_cta = (tmp[f_size]+CTA_SIZE-1)/CTA_SIZE;
//fprintf(stdout, "tot: %d %d\n", f_size, tmp[f_size]);
#endif				

				gpu_operator_kernel_4<<<num_cta, cta_size>>>(edges, frontier, output);
#endif
#ifdef TWC
				gpu_runtime::TWC_load_balance_info<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_runtime::split_frontier<gpu_runtime::AccessorSparse><<<num_cta, cta_size>>>(edges, frontier);
				gpu_runtime::swap_queues(frontier);
				gpu_runtime::TWC_load_balance_info<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_operator_kernel_2<<<num_cta, cta_size>>>(edges, frontier, output);
				gpu_runtime::TWC_load_balance_info_mid_bin<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_operator_kernel_5_mid<<<num_cta, cta_size>>>(edges, frontier, output);
				gpu_runtime::TWC_load_balance_info_large_bin<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_operator_kernel_5_large<<<num_cta, cta_size>>>(edges, frontier, output);
#endif

#ifdef TWCE
//fprintf(stderr,"oo\n");
				gpu_runtime::TWCE_load_balance_info<gpu_runtime::AccessorSparse>(frontier, num_cta, cta_size);
				gpu_operator_kernel_6<<<num_cta, cta_size>>>(edges, frontier, output);
#endif

				cudaDeviceSynchronize();
				gpu_runtime::swap_bytemaps(output);
				output.format_ready = gpu_runtime::VertexFrontier::BYTEMAP;

			}
			gpu_runtime::deleteObject(frontier);
			frontier = output;
			rounds = (rounds + 1);
			if ((rounds) == (n)) {
				std::cout << "negative cycle" << std::endl;
				break;
			}
//if(rounds == 21) break;
		}

		
		float elapsed_time = stopTimer();
		gpu_runtime::deleteObject(frontier);

		int32_t nv = gpu_runtime::builtin_getVertices(edges); 
		cudaMemcpy(__host_SP, __device_SP, sizeof(int32_t)*nv, cudaMemcpyDeviceToHost);

		char buf[300];
		strcpy(buf, argv[1]);
		strcat(buf, ".valid");
		FILE *fp = fopen(buf, "r");
		for(int i=0;i<nv;i++) {
			int dist;
			fscanf(fp, "%d", &dist);
			if(dist != __host_SP[i]) {printf("FAIL %d: %d %d\n", i, dist, __host_SP[i]); exit(0);}	
		}		
		printf("SUCC");
		fclose(fp);


		std::cout << ",";
		std::cout << elapsed_time;
		std::cout << ",";
		std::cout << rounds << std::endl;
	}
}

