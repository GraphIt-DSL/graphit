#ifndef SWARM_INTRINSICS_H
#define SWARM_INTRINSICS_H
#include <iostream>
#include <numeric>

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
/*
template <typename T1, typename T2>
bool sum_reduce(T1& dst, T2 src) {
  dst += src;
  return true;
}
*/

template <typename T>
bool min_reduce(T& dst, T src) {
  if (dst > src) {
    dst = src;
    return true;
  }
  return false;
}


// Integer logarithm utilities

// log base 2, rounded down
inline uint32_t log2_floor(uint64_t val) {
    return val ? 63 - __builtin_clzl(val) : 0;
}
// log base 2, rounded up
inline uint32_t log2_ceil(uint64_t val) {
    return (val > 1) ? log2_floor(val - 1) + 1 : 0;
}


// Stride is used in the CC pointer jump implementation.
// Stride accross a range to reduce conflicts.
template<typename IntegerType>
struct Stride {
  IntegerType smallerRange;
  IntegerType largerRange;
  unsigned nbits;
  unsigned nlsb;

  Stride(IntegerType range) {
    smallerRange = range;

    // Calculate the number of bits needed to represent one element of the range
    nbits = log2_ceil(range);
    largerRange = 1ul << nbits;
    assert(range <= largerRange);
    assert(range > largerRange/2);

    // What should be the period with which the stride wraps around?
    // A larger period means more distinct cachelines will be in the working set,
    // occupying more cache capacity but exposing more parallelism.
    uint32_t wraparound_period = 8 * swarm::num_threads();

    // Based on the number of active cache lines needed, we figure out
    // which low-order bits to swap with the high order bits.
    nlsb = log2_ceil(wraparound_period);
  }

  IntegerType operator()(IntegerType i) const {
    uint64_t mask  = (1ul << nlsb) - 1ul;
    return ((i & mask) << (nbits - nlsb)) | (i >> nlsb);
  }
};

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

template <typename T>
void deleteObject(swarm::UnorderedQueue<T> *frontier) {
	delete frontier;
}


}

#endif
