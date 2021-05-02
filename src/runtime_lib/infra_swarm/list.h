#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_

#include <vector>
#include "vertex_frontier.h"
#include "scc/queues.h"

namespace swarm_runtime {
class VertexFrontierList {
 public:
  const int32_t max_num_elems;
 private:
  char _padding_0[SWARM_CACHE_LINE - sizeof(max_num_elems)];

 public:
  int32_t current_level = -1; // Let's have this point to the last inserted frontier. If VFL is empty, then this is -1.
 private:
  char _padding_1[SWARM_CACHE_LINE - sizeof(current_level)];

 public:
  struct alignas(SWARM_CACHE_LINE) paddedUnorderedQueue {
    swarm::UnorderedQueue<int> q;
    char _padding[SWARM_CACHE_LINE - sizeof(q)];
  };
  static_assert(sizeof(paddedUnorderedQueue) == SWARM_CACHE_LINE, "");

  std::vector<paddedUnorderedQueue> frontiers;

  VertexFrontierList(int32_t max_elems) : max_num_elems(max_elems) {
    // Ensure reallocation (which is expensive) is rare, by allocation space
    // in large chunks.  Note: do NOT call frontiers.reserve() anywhere else.
    // Instead, let push_back() or emplace_back() do all the work: they will
    // double the allocated capacity when needed, ensuring that reallocations
    // stay rare.  As no application uses many instances of VertexFrontierList,
    // allocating 16k cachelines (1 MB) is acceptable.
    frontiers.reserve(16 * 1024);
  }

  void extend_frontier_list() {
    for (int i = 0; i < 10; i++) {
       frontiers.emplace_back();
    }
  }
};

VertexFrontierList create_new_vertex_frontier_list(int32_t max_elems) {
  VertexFrontierList vl(max_elems);
  vl.extend_frontier_list();
  return vl;
}

// Inserts should increment the current_level to point to the next empty frontier to either swap or insert to.
void builtin_insert(VertexFrontierList &v1, VertexFrontier &frontier) {
  v1.current_level++;
  for (int i = 0; i < frontier.elems.size(); i++) {
    v1.frontiers[v1.current_level].q.push(frontier[i]);
  }
  if (v1.current_level >= v1.frontiers.size() - 1) {
    v1.extend_frontier_list();
  }
}

template <typename T>
static void builtin_insert(VertexFrontierList &v1, swarm::UnorderedQueue<T> *frontier) {
  v1.current_level++;
  v1.frontiers[v1.current_level].q = *frontier;

  if (v1.current_level >= v1.frontiers.size() - 1) {
    v1.extend_frontier_list();
  }
}

// insert, given an explicit round to insert into.
void builtin_insert(VertexFrontierList &v1, int vertex, int round) {
  v1.frontiers[round].q.push(vertex);
}

// pop the last frontier. This is pointed to by current_level.
void builtin_retrieve(VertexFrontierList &v1, VertexFrontier &frontier) {
  if (v1.current_level < 0) {
    return;
  }
  int total = v1.frontiers[v1.current_level].q.startMaterialize();
  frontier.elems.resize(total);
  int32_t* a = frontier.elems.data();
  v1.frontiers[v1.current_level].q.finishMaterialize(total, a);
  frontier.num_elems = total;
  
  v1.current_level--;
}

// pop the last frontier, pointed to by current_level
template <typename T>
static void builtin_retrieve(VertexFrontierList &v1, swarm::UnorderedQueue<T> *frontier) {
  if (v1.current_level < 0) {
    return;
  }
  *frontier = std::move(v1.frontiers[v1.current_level].q);
  /*
  frontier->clear();
  int total = v1.frontiers[v1.current_level].startMaterialize();
  int32_t* a = new int32_t[total];
  v1.frontiers[v1.current_level].finishMaterialize(total, a);
  for (int i = 0; i < total; i++) {
    frontier->push(a[i]);
  } 
  */
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
