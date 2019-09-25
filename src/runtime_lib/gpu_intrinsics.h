#ifndef GPU_INTRINSICS_H
#define GPU_INTRINSICS_H

#include <iostream>
#include <string>

#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"
#include "infra_gpu/load_balance.h"
#include "graphit_timer.h"

namespace gpu_runtime {
template <typename T>
static bool __device__ writeMin(T *dst, T src) {
	if (*dst <= src)
		return false;
	T old_value = atomicMin(dst, src);
	bool ret = (old_value > src);
	return ret;
}
template <typename EdgeWeightType>
static int32_t builtin_getVertices(GraphT<EdgeWeightType> &graph) {
	return graph.num_vertices;
}


template <typename T>
static void deleteObject(T &t) {
	// Currently deleteObject is empty

}

}
#endif
