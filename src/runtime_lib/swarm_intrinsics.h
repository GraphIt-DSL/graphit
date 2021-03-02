#ifndef SWARM_INTRINSICS_H
#define SWARM_INTRINSICS_H
#include <iostream>

#include "infra_swarm/graph.h"
#include "infra_swarm/vertex_frontier.h"
#include "infra_swarm/list.h"

#define SWARM_FUNC_ATTRIBUTES __attribute__((noinline, swarmify, assertswarmified))
namespace swarm_runtime {
template <typename T>
bool sum_reduce(T& dst, T src) {
  dst += src;
  return true;
}

template <typename T>
int min_reduce(T& dst, T& src) {
  if (dst > src) {
    dst = src;
    return 0;
  } else if (dst < src) {
    src = dst;
    return 1;
  }
  return 2;
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
  //std::cout << t << std::endl;
}

void deleteObject(VertexFrontier &frontier) {
	std::vector<int32_t>().swap(frontier.elems);
}




}

#endif
