#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
#ifdef GEN_PYBIND_WRAPPERS
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif
WGraph edges;
int  * __restrict dist;
julienne::PriorityQueue < int  >* pq; 
template <typename APPLY_FUNC > VertexSubset<NodeID>* edgeset_apply_push_parallel_weighted_deduplicatied_from_vertexset_with_frontier(WGraph & g , VertexSubset<NodeID>* from_vertexset, APPLY_FUNC apply_func) 
{ 
    int64_t numVertices = g.num_nodes(), numEdges = g.num_edges();
    from_vertexset->toSparse();
    long m = from_vertexset->size();
    // used to generate nonzero indices to get degrees
    uintT *degrees = newA(uintT, m);
    // We probably need this when we get something that doesn't have a dense set, not sure
    // We can also write our own, the eixsting one doesn't quite work for bitvectors
    //from_vertexset->toSparse();
    {
        ligra::parallel_for_lambda((long)0, (long)m, [&] (long i) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
         });
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (g.get_flags_() == nullptr){
      g.set_flags_(new int[numVertices]());
      ligra::parallel_for_lambda(0, (int)numVertices, [&] (int i) { g.get_flags_()[i]=0; });
    }
    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    if (numVertices != from_vertexset->getVerticesRange()) {
        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    if (outDegrees == 0) return next_frontier;
    uintT *offsets = degrees;
    long outEdgeCount = sequence::plusScan(offsets, degrees, m);
    uintE *outEdges = newA(uintE, outEdgeCount);
  ligra::parallel_for_lambda((long)0, (long)m, [&] (long i) {
    NodeID s = from_vertexset->dense_vertex_set_[i];
    int j = 0;
    uintT offset = offsets[i];
    for(WNode d : g.out_neigh(s)){
      if( apply_func ( s , d.v, d.w ) && CAS(&(g.get_flags_()[d.v]), 0, 1)  ) { 
        outEdges[offset + j] = d.v; 
      } else { outEdges[offset + j] = UINT_E_MAX; }
      j++;
    } //end of for loop on neighbors
  });
  uintE *nextIndices = newA(uintE, outEdgeCount);
  long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
  free(outEdges);
  free(degrees);
  next_frontier->num_vertices_ = nextM;
  next_frontier->dense_vertex_set_ = nextIndices;
  ligra::parallel_for_lambda((int)0, (int)nextM, [&] (int i) {
     g.get_flags_()[nextIndices[i]] = 0;
  });
  return next_frontier;
} //end of edgeset apply function 
struct dist_generated_vector_op_apply_func_0
{
  void operator() (NodeID v) 
  {
    dist[v] = (2147483647) ;
  };
};
struct updateEdge
{
  bool operator() (NodeID src, NodeID dst, int weight) 
  {
    bool output3 ;
    bool dist_trackving_var_2 = (bool) 0;
    int new_dist = (dist[src] + weight);
    dist_trackving_var_2 = pq->updatePriorityMinAtomic(dst, dist[dst], new_dist);
    output3 = dist_trackving_var_2;
    return output3;
  };
};
struct printDist
{
  void operator() (NodeID v) 
  {
    std::cout << dist[v]<< std::endl;
  };
};
struct reset
{
  void operator() (NodeID v) 
  {
    dist[v] = (2147483647) ;
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadWeightedEdgesFromFile ( argv_safe((1) , argv, argc)) ;
  dist = new int [ builtin_getVertices(edges) ];
  ligra::parallel_for_lambda((int)0, (int)builtin_getVertices(edges) , [&] (int vertexsetapply_iter) {
    dist_generated_vector_op_apply_func_0()(vertexsetapply_iter);
  });;
  for ( int trail = (0) ; trail < (10) ; trail++ )
  {
    ligra::parallel_for_lambda((int)0, (int)builtin_getVertices(edges) , [&] (int vertexsetapply_iter) {
      reset()(vertexsetapply_iter);
    });;
    startTimer() ;
    int start_vertex = atoi(argv_safe((2) , argv, argc)) ;
    dist[start_vertex] = (0) ;
    pq = new julienne::PriorityQueue <int  > ( edges.num_nodes(), dist, (julienne::bucket_order)1, (julienne::priority_order)2, 128, 2);
    while ( (pq->finished() ) == ((bool) 0))
    {
      VertexSubset<int> *  frontier = getBucketWithGraphItVertexSubset(pq) ;
      VertexSubset<int> *  modified_vertexsubset1 = edgeset_apply_push_parallel_weighted_deduplicatied_from_vertexset_with_frontier(edges, frontier, updateEdge()); 
      updateBucketWithGraphItVertexSubset(modified_vertexsubset1, pq, 0, 2);
      deleteObject(frontier) ;
    }
    float elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
  }
};
#ifdef GEN_PYBIND_WRAPPERS
PYBIND11_MODULE(, m) {
}
#endif

