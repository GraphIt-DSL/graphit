#ifndef BUILTINS_GUNROCK_H
#define BUILTINS_GUNROCK_H
#include <gunrock/gunrock.h>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/test_base.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/frontier.cuh>
#include <gunrock/app/enactor_types.cuh>
#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/util/device_intrinsics.cuh>
#include <time.h>
#include <chrono>


//double atomicAdd __device__ (double* address, double val);

#if 1
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
#endif



template<typename GraphT>
size_t builtin_getVertices __device__ __host__ (GraphT &graph) {
	return graph.nodes;
}
template<typename GraphT>
size_t builtin_getEdges __device__ __host__ (GraphT &graph) {
	return graph.edges;
}
template<typename GraphT>
gunrock::util::Array1D<size_t, int> __host__ builtin_getOutDegrees(GraphT &graph) {
	gunrock::util::Array1D<size_t, int> OutDegrees;
	OutDegrees.Allocate(graph.nodes, gunrock::util::DEVICE);
	OutDegrees.Allocate(graph.nodes, gunrock::util::HOST);
	OutDegrees.ForEach(
		[] __device__ (int &v) {
			v = 0;
		}, graph.nodes, gunrock::util::DEVICE, 0);
	gunrock::oprtr::ForAll((uint32_t *)NULL, 
		[graph, OutDegrees] __device__ (uint32_t *dummy, const size_t &edge_id) {
			uint32_t src, dst;
			graph.GetEdgeSrcDest(edge_id, src, dst);
			//OutDegrees[v] = graph.row_offsets[v+1] - graph.row_offsets[v];		
			atomicAdd(&OutDegrees[src], 1);
		}, graph.edges, gunrock::util::DEVICE, 0);
	return OutDegrees;
}
template <typename T>
bool builtin_writeSum __device__ (T &src, T val) {
	if (val){
		atomicAdd(&src, val);
		return true;
	}
	return false;
}
template <typename T>
bool builtin_writeMin __device__ (T &src, T val) {
	auto old = atomicMin(&src, val);
	if (old > val)
		return true;
	return false;
}

template<typename T>
size_t builtin_getVertexSetSize(T &frontier) {
	frontier.GetQueueLength(0);
	return frontier.queue_length;
}

template<typename T, typename S>
void builtin_addVertex(T &frontier, S element){
	frontier.GetQueueLength(0);
	auto old_index = frontier.queue_length;
	frontier.queue_length += 1;
	frontier.V_Q()->operator[](old_index) = element;
	frontier.V_Q()->Move(gunrock::util::HOST, gunrock::util::DEVICE, 1, old_index, 0);
	frontier.work_progress.SetQueueLength(frontier.queue_index, frontier.queue_length);
	return;
}


template<typename T, typename S> 

void builtin_addAllVertices(T &frontier, S element_max) {
	frontier.queue_length = element_max;
	for (S c = 0; c < element_max; c ++ )
		frontier.V_Q()->operator[](c) = c;
	frontier.V_Q()->Move(gunrock::util::HOST, gunrock::util::DEVICE, element_max, 0, 0);
	frontier.work_progress.SetQueueLength(frontier.queue_index, frontier.queue_length);
}

struct timeval start_time_;
struct timeval elapsed_time_;

void startTimer(){
    gettimeofday(&start_time_, NULL);
}

float stopTimer(){
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
    return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6;

}


#endif
