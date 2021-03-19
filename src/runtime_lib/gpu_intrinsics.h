#ifndef GPU_INTRINSICS_H
#define GPU_INTRINSICS_H

#include <iostream>
#include <string>

#include "infra_gpu/graph.h"
#include "infra_gpu/vertex_frontier.h"
#include "infra_gpu/vertex_representation.h"
#include "infra_gpu/load_balance.h"
#include "graphit_timer.h"
#include "infra_gpu/support.h"
#include "infra_gpu/printer.h"
#include "infra_gpu/gpu_priority_queue.h"
#include "infra_gpu/list.h"

namespace gpu_runtime {

void deleteObject(VertexFrontier &t) {
	delete_vertex_frontier(t);
}

template <typename T>
void deleteObject(GPUPriorityQueue<T> &pq) {
	pq.release();
}

 void * no_args[1];

float str_to_float(const char* str) {
	float val;
	if (sscanf(str, "%f", &val) != 1)
		return 0.0;
	return val;
}
int32_t str_to_int(const char* str) {
	int32_t val;
	if (sscanf(str, "%i", &val) != 1)
		return 0;
	return val;
}
}
#endif
