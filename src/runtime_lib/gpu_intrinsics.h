#ifndef GPU_INTRINSICS_H
#define GPU_INTRINSICS_H

#include <iostream>
#include <string>

#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"
#include "infra_gpu/load_balance.h"
#include "timer.h"

namespace gpu_runtime {
template <typename T>
T __device__ writeMin(T *dst, T src) {
	return atomicMin(dst, src);
}
template <typename EdgeWeightType>
static int32_t builtin_getVertices(GraphT<EdgeWeightType> &graph) {
	return graph.num_vertices;
}

static VertexFrontier create_new_vertex_set(int32_t num_vertices) {
	return VertexFrontier();
}

static void builtin_addVertex(VertexFrontier &frontier, int32_t vid) {
	
}

template <void body(int32_t vid)>
static void __global__ vertex_set_apply_kernel(int32_t num_vertices) {

} 

static int32_t builtin_getVertexSetSize(VertexFrontier &frontier) {
	return 0;	
}

template <typename T>
void deleteObject(T &t) {

}

void __device__ enqueueVertexSparseQueue(int32_t *sparse_queue, int32_t *sparse_queue_size, int32_t vertex_id) {
}
}
#endif
