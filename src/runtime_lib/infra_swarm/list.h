#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_

#include <vector>
#include "vertex_frontier.h"
#include "scc/queues.h"

namespace swarm_runtime {
class VertexFrontierList {
 public:
  int32_t max_num_elems;
  int32_t current_level;

  std::vector<swarm::UnorderedQueue<int>> frontiers;

  void extend_frontier_list() {
    for (int i = 0; i < 10; i++) {
      frontiers.emplace_back();
    }
  }
};

VertexFrontierList create_new_vertex_frontier_list(int32_t max_elems) {
  VertexFrontierList vl;
  vl.max_num_elems = max_elems;
  vl.current_level = 0;

  vl.extend_frontier_list();
  return vl;
}


void builtin_insert(VertexFrontierList &v1, VertexFrontier &frontier) {
  for (int i = 0; i < frontier.elems.size(); i++) {
    v1.frontiers[v1.current_level].push(frontier[i]);
  }
  v1.current_level++;
  if (v1.current_level >= v1.frontiers.size()) {
    v1.extend_frontier_list();
  }
}

// insert, given an explicit round to insert into.
void builtin_insert(VertexFrontierList &v1, int vertex, int round) {
  v1.frontiers[round].push(vertex);
  if (round >= v1.frontiers.size()) v1.extend_frontier_list();
}

// pop the last frontier
void builtin_retrieve(VertexFrontierList &v1, VertexFrontier &frontier) {
  int32_t round = v1.frontiers.size() - 1;
  int32_t* a = &frontier.elems[0];
  int total = v1.frontiers[round].materialize(a);
  v1.current_level--;
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
