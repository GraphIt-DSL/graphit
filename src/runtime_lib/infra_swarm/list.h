#ifndef GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
#define GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_

#include <vector>
namespace swarm_runtime {
class VertexFrontierList {
 public:
  int32_t max_num_elems;
  int32_t current_level;

  std::vector<int32_t> frontier_separator_idxs;
  std::vector<int32_t> vertices;
};

VertexFrontierList create_new_vertex_frontier_list(int32_t max_elems) {
  VertexFrontierList vl;
  vl.max_num_elems = max_elems;
  vl.current_level = 0;

  return vl;
}
void builtin_insert(VertexFrontierList &vl, VertexFrontier &frontier) {
  for (int i = 0; i < frontier.elems.size(); i++) {
    v1.vertices.push_back(frontier.elems[i]);
  }
  frontier_separator_idxs.push_back(frontier.elems.size());
  v1.current_level++;
}

void builtin_insert(VertexFrontierList &vl, int vertex) {
  v1.vertices.push_back(vertex);
}

void builtin_retrieve(VertexFrontierList &vl, VertexFrontier &frontier) {
  if (frontier.elems.size() > 0) {
    vector<int32_t>().swap(frontier.elems); // empty out the existing frontier
  }
  int32_t start_idx = 0;
  int32_t end_idx = v1.vertices.size();
  if (v1.current_level > 0) start_idx = v1.frontier_separator_idxs[v1.current_level - 1];
  if (v1.current_level > 0 && v1.current_level < v1.frontier_separator_idxs.size()) end_idx = v1.frontier_separator_idxs[v1.current_level];
  for (int32_t i = start_idx; i < end_idx; i++) {
    frontier.elems.push_back(v1.vertices[i]);
  }
  v1.current_level--;
}

void builtin_increment_round(VertexFrontierList &vl) {
  int32_t v_size = v1.vertices.size();
  v1.frontier_separator_idxs.push_back(v_size);
  v1.current_level++;
}

}
#endif //GRAPHIT_SRC_RUNTIME_LIB_INFRA_SWARM_LIST_H_
