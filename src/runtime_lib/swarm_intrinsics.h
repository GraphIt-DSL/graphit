#ifndef SWARM_INTRINSICS_H
#define SWARM_INTRINSICS_H
#include <iostream>

#include "infra_swarm/graph.h"
#include "infra_swarm/vertex_frontier.h"
#include "infra_swarm/list.h"
#include "scc/opt.h"

#define SWARM_FUNC_ATTRIBUTES __attribute__((noinline, assertswarmified))
namespace swarm_runtime {

template <typename T1, typename T2>
bool sum_reduce(T1& dst, T2 src) {
  SCC_OPT_TASK_CACHELINEHINT(&dst, {
    dst += src;
    });
  return true;
}

template <typename T>
bool min_reduce(T& dst, T& src) {
  if (dst > src) {
    dst = src;
    return true;
  }
  return false;
}

void startTimer() {
  // Currently left empty
  // Insert appropriate timing code for swarm
}


float stopTimer() {
  // stop and reset timer
  // returns time elapsed in seconds since the last startTimer
  return 0;
}

template <typename T>
void print(T& t) {
	std::cout << t << std::endl;
}

void deleteObject(VertexFrontier &frontier) {
	frontier.elems.clear();
	frontier.elems.shrink_to_fit();
//	std::vector<int32_t>().swap(frontier.elems);
	frontier.num_elems = 0;
}




}

#endif
