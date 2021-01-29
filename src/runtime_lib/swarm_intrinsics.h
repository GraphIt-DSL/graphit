#ifndef SWARM_INTRINSICS_H
#define SWARM_INTRINSICS_H
#include <iostream>

#include "pls/pls_api.h"
#include "pls/algorithm.h"
#include "thread_local_queues.h"

#include "infra_swarm/graph.h"
#define SWARM_FUNC_ATTRIBUTES __attribute__((noinline, swarmify, assertswarmified))
namespace swarm_runtime {
template <typename T>
bool sum_reduce(T& dst, T src) {
  dst += src;
  return true;
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



}

#endif