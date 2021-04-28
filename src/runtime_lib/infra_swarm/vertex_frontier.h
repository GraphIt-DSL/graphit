#include "scc/queues.h"
//#include <scc/autoparallel.h>
#include <vector>

#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
namespace swarm_runtime {

class VertexFrontier {
 public:
  std::vector<int32_t> elems;
  int num_elems = 0;
  int& operator[](int);
  int size() {
    return num_elems;
  }
};

static VertexFrontier create_new_vertex_set(int32_t num_vertices, int32_t init_elems = 0) {
  VertexFrontier frontier;
  if (init_elems == num_vertices) {
    for (int i = 0, e = init_elems; i < e; i++) {
      frontier.elems.push_back(i);
    }
    frontier.num_elems = init_elems;
  }
  return frontier;
}

// this is chaotic
int& VertexFrontier::operator[](int idx) {
  if (idx >= num_elems) {
	  std::cout << "Vertex frontier out of bounds.";
    	  exit(0);
  }
  return elems[idx];
}

template <typename T>
static void builtin_addVertex(swarm::UnorderedQueue<T> *frontier, int32_t vid) {
  frontier->push(vid);
}

static void builtin_addVertex(VertexFrontier &frontier, int32_t vid) {
  frontier.elems.push_back(vid);
  frontier.num_elems++;
}

static void clear_frontier(VertexFrontier &frontier) {
  frontier.elems.clear();
  frontier.num_elems = 0;
	//std::vector<int32_t>().swap(frontier.elems);
}

template <typename T>
static void clear_frontier(swarm::UnorderedQueue<T> *frontier) {
  frontier->clear();
}

//template <typename SwarmQueueType>
//static void builtin_addVertex(VertexFrontier &frontier, SwarmQueueType &swarm_queue, int32_t vid) {
//  frontier.elems.push_back(vid);
//  swarm_queue.push_init(vid);
//}

template <typename T>
static int32_t builtin_getVertexSetSize(swarm::UnorderedQueue<T> *frontier) {
  if (frontier->empty()) return 0;
  return 1;
}

template <typename SwarmQueueType>
static void populate_swarm_frontier(VertexFrontier &frontier, SwarmQueueType &swarm_queue, int tuple_elems) {
  for (auto elem : frontier.elems) {
    swarm_queue.push_init(0, elem);
  }
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
