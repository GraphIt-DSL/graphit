#ifndef SWARM_GRAPH_H
#define SWARM_GRAPH_H

// GraphT data structure 
#define IGNORE_JULIENNE_TYPES
#include "infra_gapbs/benchmark.h"

namespace swarm_runtime {
template <typename EdgeWeightType>
class GraphT {
 public:
  int num_vertices;
  int num_edges;

  int* src_offsets;
  int* edge_src;
  int* edge_dst;

  EdgeWeightType * edge_weight;

  int * out_degrees;
};

template <typename EdgeWeightType>
int builtin_getVertices(GraphT<EdgeWeightType> &graph) {
  return graph.num_vertices;
}

template <typename EdgeWeightType>
int* builtin_getOutDegrees(GraphT<EdgeWeightType> &graph) {
  if (graph.out_degrees == nullptr) {
    graph.out_degrees = new int[graph.num_vertices];
    for (int i = 0; i < graph.num_vertices; i++) {
      graph.out_degrees[i] = graph.src_offsets[i+1] - graph.src_offsets[i];
    }
  }
  return graph.out_degrees;
}

template <typename EdgeWeightType>
void load_graph(GraphT<EdgeWeightType> &graph, std::string filename) {
  CLBase cli (filename);
  WeightedBuilder builder(cli);
  WGraph g = builder.MakeGraph();
  graph.num_vertices = g.num_nodes();
  graph.num_edges = g.num_edges();

  graph.src_offsets = new int[graph.num_vertices + 1];
  graph.edge_src = new int[graph.num_edges];
  graph.edge_dst = new int[graph.num_edges];
  graph.edge_weight = new EdgeWeightType[graph.num_edges];
  int tmp = 0;
  graph.src_offsets[0] = tmp;
  for (int i = 0; i < graph.num_vertices; i++) {
    for (auto j: g.out_neigh(i)) {
      graph.edge_src[tmp] = i;
      graph.edge_dst[tmp] = j.v;
      graph.edge_weight[tmp] = j.w;
      tmp++;
    }
    graph.src_offsets[i+1] = tmp;
  }
  builtin_getOutDegrees(graph);
}

}

#endif