#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_

#include <vector>
#include "vertex_frontier.h"

namespace swarm_runtime {
class VertexFrontierList {
 public:
  int32_t max_num_elems;
  int32_t current_level;

  std::vector<std::vector<int32_t>> frontiers;

  void extend_frontier_list() {
    for (int i = 0; i < 10; i++) {
      std::vector<int32_t> new_frontier;
      frontiers.push_back(new_frontier);
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
    v1.frontiers[v1.current_level].push_back(frontier[i]);
  }
  v1.current_level++;
  if (v1.current_level >= v1.frontiers.size()) {
    v1.extend_frontier_list();
  }
}

void builtin_insert(VertexFrontierList &v1, int vertex, int round) {
  v1.frontiers[round].push_back(vertex);
  if (round >= v1.frontiers.size()) v1.extend_frontier_list();
}

void builtin_retrieve(VertexFrontierList &v1, VertexFrontier &frontier) {
  if (frontier.elems.size() > 0) {
	  std::vector<int32_t>().swap(frontier.elems); // empty out the existing frontier
  }
  int32_t round = v1.frontiers.size() - 1;
  for (int32_t i : v1.frontiers[round]) {
    frontier.elems.push_back(i);
  }
  v1.current_level--;
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
