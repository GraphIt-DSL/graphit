#include "scc/queues.h"

#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
namespace swarm_runtime {

class VertexFrontier {
 public:
  std::vector<int32_t> elems;
  swarm::PrioQueue<int32_t> swarm_frontier;

  int& operator[](int);
  int size() {
    return elems.size();
  }
};

// this is chaotic
int& VertexFrontier::operator[](int idx) {
  if (idx >= elems.size()) {
    cout << "Vertex frontier out of bounds.";
    exit(0);
  }
  return elems[idx];
}

static void builtin_addVertex(VertexFrontier &frontier, int32_t vid) {
  frontier.elems.push_back(vid);
  frontier.swarm_frontier.push_init(0, vid);
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_VERTEX_FRONTIER_H_
