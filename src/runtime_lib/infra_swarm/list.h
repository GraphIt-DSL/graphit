#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_

#include <vector>
#include "vertex_frontier.h"
#include "scc/queues.h"

namespace swarm_runtime {
class VertexFrontierList {
 public:
  int32_t max_num_elems;
  int32_t current_level;   // Let's have this point to the last inserted frontier. If VFL is empty, then this is -1.

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
  vl.current_level = -1; // Points at the current head. If there is one frontier in the VFL, then this will be 0.
  vl.extend_frontier_list();
  return vl;
}

// Inserts should increment the current_level to point to the next empty frontier to either swap or insert to.
void builtin_insert(VertexFrontierList &v1, VertexFrontier &frontier) {
  v1.current_level++;
  for (int i = 0; i < frontier.elems.size(); i++) {
    v1.frontiers[v1.current_level].push(frontier[i]);
  }
  if (v1.current_level >= v1.frontiers.size() - 1) {
    v1.extend_frontier_list();
  }
}

template <typename T>
static void builtin_insert(VertexFrontierList &v1, swarm::UnorderedQueue<T> *frontier) {
  v1.current_level++;
  v1.frontiers[v1.current_level] = *frontier;
}

// insert, given an explicit round to insert into.
void builtin_insert(VertexFrontierList &v1, int vertex, int round) {
  v1.frontiers[round].push(vertex);
}

// pop the last frontier. This is pointed to by current_level.
void builtin_retrieve(VertexFrontierList &v1, VertexFrontier &frontier) {
  if (v1.current_level < 0) {
    return;
  }
  int total = v1.frontiers[v1.current_level].startMaterialize();
  frontier.elems.resize(total);
  int32_t* a = frontier.elems.data();
  v1.frontiers[v1.current_level].finishMaterialize(total, a);
  frontier.num_elems = total;
  
  v1.current_level--;
}

// pop the last frontier, pointed to by current_level
template <typename T>
static void builtin_retrieve(VertexFrontierList &v1, swarm::UnorderedQueue<T> *frontier) {
  if (v1.current_level < 0) {
    return;
  }
  *frontier = std::move(v1.frontiers[v1.current_level]);
  v1.current_level--;
}

void builtin_update_size(VertexFrontierList &v1, int new_head) {
  v1.current_level = new_head - 1;  // new_head is a pointer to the next frontier to insert into. Since this doesn't really exist per say as a frontier, we set current_level to the one right before that.
  if (new_head >= v1.frontiers.size() - 1) v1.extend_frontier_list();
}

// v1.current_level points to the current head. So we have to add 1 to get the actual size.
// For example, if there were two frontiers, then current_level would be 1 (frontier 0 and frontier 1).
int builtin_get_size(VertexFrontierList &v1) {
  return v1.current_level + 1;
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
