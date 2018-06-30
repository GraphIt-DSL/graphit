#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
Graph edges; 
int  * __restrict parent;
template <typename TO_FUNC , typename APPLY_FUNC, typename PUSH_APPLY_FUNC> VertexSubset<NodeID>* edgeset_apply_hybrid_dense_parallel_from_vertexset_to_filter_func_with_frontier_pull_frontier_bitvector(Graph & g , VertexSubset<NodeID>* from_vertexset, TO_FUNC to_func, APPLY_FUNC apply_func, PUSH_APPLY_FUNC push_apply_func) 
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
  Bitmap bitmap(numVertices);
  bitmap.reset();
  parallel_for(int i = 0; i < numVertices; i+=32){
     int start = i;
     int end = (((i + 32) < numVertices)? (i+32):numVertices);
     for(int j = start; j < end; j++){
        if (from_vertexset->bool_map_[j])
          bitmap.set_bit(j);
     }
  }
    parallel_for ( NodeID d=0; d < g.num_nodes(); d++) {
      if (to_func(d)){ 
        for(NodeID s : g.in_neigh(d)){
          if (bitmap.get_bit(s)) { 
            if( apply_func ( s , d ) ) { 
              next[d] = 1; 
              if (!to_func(d)) break; 
            }
          }
        } //end of loop on in neighbors
      } //end of to filtering 
    } //end of outer for loop
  next_frontier->num_vertices_ = sequence::sum(next, numVertices);
  free(next_frontier->bool_map_);
  next_frontier->bool_map_ = next;
  free(degrees);
  return next_frontier;
} else {
    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);
    if (numVertices != from_vertexset->getVerticesRange()) {
        cout << "edgeMap: Sizes Don't match" << endl;
        abort();
    }
    if (outDegrees == 0) {
      free(degrees);
      return next_frontier;
    }
    uintT *offsets = degrees;
    long outEdgeCount = sequence::plusScan(offsets, degrees, m);
    uintE *outEdges = newA(uintE, outEdgeCount);
      parallel_for (long i=0; i < m; i++) {
    NodeID s = from_vertexset->dense_vertex_set_[i];
    int j = 0;
    uintT offset = offsets[i];
        for(NodeID d : g.out_neigh(s)){
          if (to_func(d)) { 
            if( push_apply_func ( s , d  ) ) { 
              outEdges[offset + j] = d; 
            } else { outEdges[offset + j] = UINT_E_MAX; }
          } //end of to func
           else { outEdges[offset + j] = UINT_E_MAX;  }
          j++;
        } //end of for loop on neighbors
      }
  uintE *nextIndices = newA(uintE, outEdgeCount);
  long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());
  free(outEdges);
  free(degrees);
  next_frontier->num_vertices_ = nextM;
  delete[] next_frontier->dense_vertex_set_;
  next_frontier->dense_vertex_set_ = nextIndices;
  return next_frontier;
} //end of else
} //end of edgeset apply function 
struct updateEdge_push_ver
{
  bool operator() (NodeID src, NodeID dst) 
  {
    bool output3 ;
    bool parent_trackving_var_2 = (bool) 0;
    parent_trackving_var_2 = compare_and_swap ( parent[dst],  -(1) , src);
    output3 = parent_trackving_var_2;
    return output3;
  };
};
struct parent_generated_vector_op_apply_func_0
{
  void operator() (NodeID v) 
  {
    parent[v] =  -(1) ;
  };
};
struct updateEdge
{
  bool operator() (NodeID src, NodeID dst) 
  {
    bool output1 ;
    parent[dst] = src;
    output1 = (bool) 1;
    return output1;
  };
};
struct toFilter
{
  bool operator() (NodeID v) 
  {
    bool output ;
    output = (parent[v]) == ( -(1) );
    return output;
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadEdgesFromFile ( argv[(1) ]) ;
  edges.buildPushSegmentedGraphs("s1", atoi(argv[3]));
  parent = new int [ builtin_getVertices(edges) ];
  for ( int trail = (0) ; trail < (10) ; trail++ )
  {
    parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
      parent_generated_vector_op_apply_func_0()(i);
    };
    startTimer() ;
    VertexSubset<int> *  frontier = new VertexSubset<int> ( builtin_getVertices(edges)  , (0) );
    builtin_addVertex(frontier, atoi(argv[2]) ) ;
    parent[atoi(argv[2]) ] = atoi(argv[2]) ;
    while ( (builtin_getVertexSetSize(frontier) ) != ((0) ))
      {
	frontier = edgeset_apply_hybrid_dense_parallel_from_vertexset_to_filter_func_with_frontier_pull_frontier_bitvector(edges, frontier, toFilter(), updateEdge(), updateEdge_push_ver()); 
      }
    float elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
  }
};

