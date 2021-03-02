#include "scc/queues.h"

#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
namespace swarm_runtime {

class VertexFrontier {
 public:
  std::vector<int32_t> elems;

  int& operator[](int);
  int size() {
    return elems.size();
  }
};

static VertexFrontier create_new_vertex_set(int32_t num_vertices, int32_t init_elems = 0) {
  VertexFrontier frontier;
  return frontier;
}

// this is chaotic
int& VertexFrontier::operator[](int idx) {
  if (idx >= elems.size()) {
	  std::cout << "Vertex frontier out of bounds.";
    	  exit(0);
  }
  return elems[idx];
}

static void builtin_addVertex(VertexFrontier &frontier, int32_t vid) {
  frontier.elems.push_back(vid);
}

static void clear_frontier(VertexFrontier &frontier) {
  frontier.elems.clear();
}

//template <typename SwarmQueueType>
//static void builtin_addVertex(VertexFrontier &frontier, SwarmQueueType &swarm_queue, int32_t vid) {
//  frontier.elems.push_back(vid);
//  swarm_queue.push_init(vid);
//}

template <typename SwarmQueueType>
static void populate_swarm_frontier(VertexFrontier &frontier, SwarmQueueType &swarm_queue) {
  for (auto elem : frontier.elems) {
    swarm_queue.push_init(0, elem);
  }
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
