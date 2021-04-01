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
    for (int i = 0; i < 100; i++) {
      frontiers.emplace_back();
    }
  }
};

VertexFrontierList create_new_vertex_frontier_list(int32_t max_elems) {
  VertexFrontierList vl;
  vl.max_num_elems = max_elems;
  vl.current_level = -1; // Points at the current head. If there is one frontier in the VFL, then this will be 0.

  vl.extend_frontier_list();
  return vl;
}


void builtin_insert(VertexFrontierList &v1, VertexFrontier &frontier) {
  v1.current_level++;
  for (int i = 0; i < frontier.elems.size(); i++) {
    v1.frontiers[v1.current_level].push(frontier[i]);
  }
  if (v1.current_level >= v1.frontiers.size() - 1) {
    v1.extend_frontier_list();
  }
}

// insert, given an explicit round to insert into.
void builtin_insert(VertexFrontierList &v1, int vertex, int round) {
  v1.frontiers[round].push(vertex);
}

// pop the last frontier
void builtin_retrieve(VertexFrontierList &v1, VertexFrontier &frontier) {
  //printf("v1.current_level = %d\n", v1.current_level);
  v1.current_level--;
  
  if (v1.current_level < 0) {
    return;
  }
  int total = v1.frontiers[v1.current_level].startMaterialize();
  frontier.elems.resize(total);
  int32_t* a = frontier.elems.data();
  v1.frontiers[v1.current_level].finishMaterialize(total, a);
  frontier.num_elems = total;
  
  //printf("Decremented current level: v1.current_level = %d\n", v1.current_level);
}

void builtin_update_size(VertexFrontierList &v1, int new_head) {
  v1.current_level = new_head;
  if (new_head >= v1.frontiers.size() - 1) v1.extend_frontier_list();
}

// v1.current_level points to the current head. So we have to add 1 to get the actual size.
int builtin_get_size(VertexFrontierList &v1) {
  return v1.current_level + 1;
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
