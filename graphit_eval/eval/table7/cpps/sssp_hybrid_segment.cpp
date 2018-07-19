#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
WGraph edges;
int  * __restrict SP;
template <typename APPLY_FUNC , typename PUSH_APPLY_FUNC> VertexSubset<NodeID>* edgeset_apply_hybrid_dense_parallel_weighted_deduplicatied_from_vertexset_with_frontier(WGraph & g , VertexSubset<NodeID>* from_vertexset, APPLY_FUNC apply_func, PUSH_APPLY_FUNC push_apply_func) 
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
      for (int segmentId = 0; segmentId < g.getNumSegments("s1"); segmentId++) {
      auto sg = g.getSegmentedGraph(std::string("s1"), segmentId);
parallel_for ( NodeID localId=0; localId < sg->numVertices; localId++) {
      NodeID d = sg->graphId[localId];
      for (int64_t ngh = sg->vertexArray[localId]; ngh < sg->vertexArray[localId+1]; ngh++) {
        WNode s = sg->edgeArray[ngh];
        if (from_vertexset->bool_map_[s.v] ) { 
          if( apply_func ( s.v , d, s.w ) ) { 
            next[d] = 1; 
          }
        }
      } //end of loop on in neighbors
    } //end of outer for loop
    } // end of segment for loop
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
        for(WNode d : g.out_neigh(s)){
          if( push_apply_func ( s , d.v, d.w ) && CAS(&(g.flags_[d.v]), 0, 1)  ) { 
            outEdges[offset + j] = d.v; 
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
struct updateEdge_push_ver
{
  bool operator() (NodeID src, NodeID dst, int weight) 
  {
    bool output4 ;
    bool SP_trackving_var_3 = (bool) 0;
    SP_trackving_var_3 = writeMin( &SP[dst], (SP[src] + weight) ); 
    output4 = SP_trackving_var_3;
    return output4;
  };
};
struct SP_generated_vector_op_apply_func_0
{
  void operator() (NodeID v) 
  {
    SP[v] = (2147483647) ;
  };
};
struct updateEdge
{
  bool operator() (NodeID src, NodeID dst, int weight) 
  {
    bool output2 ;
    bool SP_trackving_var_1 = (bool) 0;
    if ( ( SP[dst]) > ( (SP[src] + weight)) ) { 
      SP[dst]= (SP[src] + weight); 
      SP_trackving_var_1 = true ; 
    } 
    output2 = SP_trackving_var_1;
    return output2;
  };
};
struct reset
{
  void operator() (NodeID v) 
  {
    SP[v] = (2147483647) ;
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadWeightedEdgesFromFile ( argv[(1) ]) ;
  edges.buildPullSegmentedGraphs("s1", 15);
  SP = new int [ builtin_getVertices(edges) ];
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    SP_generated_vector_op_apply_func_0()(vertexsetapply_iter);
  };
  for ( int trail = (0) ; trail < (10) ; trail++ )
  {
    parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
      reset()(vertexsetapply_iter);
    };
    startTimer() ;
    int n = builtin_getVertices(edges) ;
    VertexSubset<int> *  frontier = new VertexSubset<int> ( builtin_getVertices(edges)  , (0) );
    builtin_addVertex(frontier, (14) ) ;
    SP[(14) ] = (14) ;
    int rounds = (0) ;
    while ( (builtin_getVertexSetSize(frontier) ) != ((0) ))
    {
      frontier = edgeset_apply_hybrid_dense_parallel_weighted_deduplicatied_from_vertexset_with_frontier(edges, frontier, updateEdge(), updateEdge_push_ver()); 
      rounds = (rounds + (1) );
      if ((rounds) == (n))
       { 
        std::cout << "negative cycle"<< std::endl;
        break;
       } 
    }
    float elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
    std::cout << "rounds"<< std::endl;
    std::cout << rounds<< std::endl;
  }
};

