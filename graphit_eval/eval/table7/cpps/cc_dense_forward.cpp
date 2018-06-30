#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
Graph edges; 
int  * __restrict IDs;
template <typename APPLY_FUNC > VertexSubset<NodeID>* edgeset_apply_hybrid_denseforward_parallel_deduplicatied_from_vertexset_with_frontier(Graph & g , VertexSubset<NodeID>* from_vertexset, APPLY_FUNC apply_func) 
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
        parallel_for (long i = 0; i < m; i++) {
            NodeID v = from_vertexset->dense_vertex_set_[i];
            degrees[i] = g.out_degree(v);
        }
    }
    uintT outDegrees = sequence::plusReduce(degrees, m);
    if (m + outDegrees > numEdges / 20) {
  VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
  bool * next = newA(bool, g.num_nodes());
  parallel_for (int i = 0; i < numVertices; i++)next[i] = 0;
    from_vertexset->toDense();
    parallel_for ( NodeID s=0; s < g.num_nodes(); s++) {
        if (from_vertexset->bool_map_[s] ) { 
          for(NodeID d : g.out_neigh(s)){
            if(  apply_func ( s , d  ) ) { 
              next[d] = 1; 
            }
          } // end of inner for loop
        } // end of if for from func or from vertexset
      } //end of outer for loop
  next_frontier->num_vertices_ = sequence::sum(next, numVertices);
  next_frontier->bool_map_ = next;
  return next_frontier;
} else {
    if (g.flags_ == nullptr){
      g.flags_ = new int[numVertices]();
      parallel_for(int i = 0; i < numVertices; i++) g.flags_[i]=0;
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
        parallel_for (long i=0; i < m; i++) {
    NodeID s = from_vertexset->dense_vertex_set_[i];
    int j = 0;
    uintT offset = offsets[i];
          for(NodeID d : g.out_neigh(s)){
            if( apply_func ( s , d  ) && CAS(&(g.flags_[d]), 0, 1)  ) { 
              outEdges[offset + j] = d; 
            } else { outEdges[offset + j] = UINT_E_MAX; }
            j++;
          } //end of for loop on neighbors
        }
  uintE *nextIndices = newA(uintE, outEdgeCount);
  long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
  free(outEdges);
  free(degrees);
  next_frontier->num_vertices_ = nextM;
  next_frontier->dense_vertex_set_ = nextIndices;
  parallel_for(int i = 0; i < nextM; i++){
     g.flags_[nextIndices[i]] = 0;
  }
  return next_frontier;
} //end of else
} //end of edgeset apply function 
struct IDs_generated_vector_op_apply_func_0
{
  void operator() (NodeID v) 
  {
    IDs[v] = (1) ;
  };
};
struct updateEdge
{
  bool operator() (NodeID src, NodeID dst) 
  {
    bool output2 ;
    bool IDs_trackving_var_1 = (bool) 0;
    IDs_trackving_var_1 = writeMin( &IDs[dst], IDs[src] ); 
    output2 = IDs_trackving_var_1;
    return output2;
  };
};
struct init
{
  void operator() (NodeID v) 
  {
    IDs[v] = v;
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadEdgesFromFile ( argv[(1) ]) ;
  IDs = new int [ builtin_getVertices(edges) ];
  for (int trail = 0; trail < 5; trail++){
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    IDs_generated_vector_op_apply_func_0()(vertexsetapply_iter);
  };
  int n = builtin_getVertices(edges) ;
  int num_iter = 0;
  startTimer() ;
  VertexSubset<int> *  frontier = new VertexSubset<int> ( builtin_getVertices(edges)  , n);
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    init()(vertexsetapply_iter);
  };
  while ( (builtin_getVertexSetSize(frontier) ) != ((0) ))
  {
    num_iter++;
    frontier = edgeset_apply_hybrid_denseforward_parallel_deduplicatied_from_vertexset_with_frontier(edges, frontier, updateEdge()); 
  }
  float elapsed_time = stopTimer() ;
  std::cout << "elapsed time: "<< std::endl;
  std::cout << elapsed_time<< std::endl;
  std::cout << "num_iter: " << num_iter << std::endl;
  }
};

