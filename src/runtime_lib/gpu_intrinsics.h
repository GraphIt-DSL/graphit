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

namespace gpu_runtime {

template <typename T>
static void deleteObject(T &t) {
	// Currently deleteObject is empty

}
template <typename T>
static __device__ void device_deleteObject(T &t) {
	// Currently deleteObject is empty
}

static void * no_args[1];
}
#endif
