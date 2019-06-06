//
// Created by Xinyi Chen on 5/28/19.
//

#ifndef GRAPHIT_GRAPH_RELABEL_H
#define GRAPHIT_GRAPH_RELABEL_H

#include <cinttypes>
#include <iostream>
#include <assert.h>

#include "builder.h"
#include "graph.h"
#include "benchmark.h"


// Relabels (and rebuilds) directed graph by order of decreasing indegree
//Takes as input a directed graph, a pointer to the source node, and a pointer to the destination node
//Returns a relabeled graph
template <typename NodeID_ = int32_t, typename WeightT_ = NodeID_, typename DestID_ = NodeID_, bool invert = true>
static
CSRGraph<NodeID_, NodeWeight<DestID_, WeightT_>, invert> relabelByIndegree(
    const CSRGraph<NodeID_, NodeWeight<DestID_, WeightT_>, invert> &g, pvector<NodeID_> &new_ids) {
  if (!g.directed()) {
    cout << "Cannot relabel undirected graph" << endl;
    std::exit(-11);
  }  
  assert(new_ids.size() == g.num_nodes());
  Timer t;
  t.Start();
  typedef std::pair<int64_t, NodeID_> degree_node_p;
  pvector<degree_node_p> degree_id_pairs(g.num_nodes());
  #pragma omp parallel for
  for (NodeID_ n=0; n < g.num_nodes(); n++)
    degree_id_pairs[n] = std::make_pair(g.in_degree(n), n);
  std::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
            std::greater<degree_node_p>());
  pvector<NodeID_> out_degrees(g.num_nodes());
  pvector<NodeID_> in_degrees(g.num_nodes());
  #pragma omp parallel for
  for (NodeID_ n=0; n < g.num_nodes(); n++) {
    out_degrees[n] = g.out_degree(degree_id_pairs[n].second);
    in_degrees[n] = g.in_degree(degree_id_pairs[n].second);
    new_ids[degree_id_pairs[n].second] = n;
  }  
  pvector<SGOffset> out_offsets = BuilderBase<NodeID_, DestID_, WeightT, invert>::ParallelPrefixSum(out_degrees);
  NodeWeight<DestID_, WeightT_>* outWeightedNeighs = new NodeWeight<DestID_, WeightT_>[out_offsets[g.num_nodes()]];
  NodeWeight<DestID_, WeightT_>** out_index = CSRGraph<NodeID_, NodeWeight<DestID_, WeightT_>>::GenIndex(out_offsets, outWeightedNeighs);
  pvector<SGOffset> in_offsets = BuilderBase<NodeID_, DestID_, WeightT, invert>::ParallelPrefixSum(in_degrees);
  NodeWeight<DestID_, WeightT_>* inWeightedNeighs = new NodeWeight<DestID_, WeightT_>[in_offsets[g.num_nodes()]];
  NodeWeight<DestID_, WeightT_>** in_index = CSRGraph<NodeID_, NodeWeight<DestID_, WeightT_>>::GenIndex(in_offsets, inWeightedNeighs);
  #pragma omp parallel for
  for (NodeID_ u=0; u < g.num_nodes(); u++) {
    // std::cout << "@@@@: " << u << std::endl;
    for (auto v : g.out_neigh(u)) {
     // std::cout << "new_ids: " << new_ids[u] << std::endl;
     // std::cout << "num nodes: " << g.num_nodes() << std::endl;
     // std::cout << "offsets: " << offsets[new_ids[u]] << std::endl;
     outWeightedNeighs[out_offsets[new_ids[u]]++] = {new_ids[v.v], v.w};
    }
    for (auto v : g.in_neigh(u)) {
     // std::cin << "new_ids: " << new_ids[u] << std::endl;
     // std::cin << "num nodes: " << g.num_nodes() << std::endl;
     // std::cin << "offsets: " << offsets[new_ids[u]] << std::endl;
     inWeightedNeighs[in_offsets[new_ids[u]]++] = {new_ids[v.v], v.w};
    }
    std::sort(out_index[new_ids[u]], out_index[new_ids[u]+1]);
    std::sort(in_index[new_ids[u]], in_index[new_ids[u]+1]);
  }
  t.Stop();
  PrintTime("Relabel", t.Seconds());
  auto relabeled_graph = CSRGraph<NodeID_, NodeWeight<DestID_, WeightT_>, invert>(g.num_nodes(), out_index, outWeightedNeighs, in_index, inWeightedNeighs);
  return relabeled_graph;
}
#endif //GRAPHIT_GRAPH_RELABEL_H
