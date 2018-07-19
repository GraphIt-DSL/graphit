#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
WGraph edges;
int  * __restrict SP;
template <typename APPLY_FUNC > VertexSubset<NodeID>* edgeset_apply_hybrid_denseforward_parallel_weighted_deduplicatied_from_vertexset_with_frontier(WGraph & g , VertexSubset<NodeID>* from_vertexset, APPLY_FUNC apply_func) 
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
      

    // parallel_for ( NodeID s=0; s < g.num_nodes(); s++) {
    //     if (from_vertexset->bool_map_[s] ) { 
    //       for(WNode d : g.out_neigh(s)){
    //         if(  apply_func ( s , d.v, d.w ) ) { 
    //           next[d.v] = 1; 
    //         }
    //       } // end of inner for loop
    //     } // end of if for from func or from vertexset
    //   } //end of outer for loop
    int numPlaces = omp_get_num_places();
    int numSegments = g.getNumSegments("s1");
    int segmentsPerSocket = (numSegments + numPlaces - 1) / numPlaces;
#pragma omp parallel num_threads(numPlaces) proc_bind(spread)
    {
      int socketId = omp_get_place_num();
      for (int i = 0; i < segmentsPerSocket; i++) {
	int segmentId = socketId + i * numPlaces;
	if (segmentId >= numSegments) break;
	auto sg = g.getSegmentedGraph(std::string("s1"), segmentId);
#pragma omp parallel num_threads(omp_get_place_num_procs(socketId)) proc_bind(close)
	{
#pragma omp for schedule(dynamic, 1024)
	  for ( NodeID localId=0; localId < sg->numVertices; localId++) {
	    NodeID s = sg->graphId[localId];
	    if (from_vertexset->bool_map_[s] ) { 
	      for (int64_t ngh = sg->vertexArray[localId]; ngh < sg->vertexArray[localId+1]; ngh++) {
		WNode d = sg->edgeArray[ngh];
		if(  apply_func ( s , d.v, d.w ) ) {
		  next[d.v] = 1;
		}
	      } //end of loop on in neighbors
	    } //end of outer for loop
	  } // end of per-socket parallel_for
	} // end of segment for loop
      }// end of per-socket parallel region
    }

  // for (int segmentId = 0; segmentId < g.getNumSegments("s1"); segmentId++) {
  //   auto sg = g.getSegmentedGraph(std::string("s1"), segmentId);
  //   parallel_for ( NodeID localId=0; localId < sg->numVertices; localId++) {
  //     NodeID s = sg->graphId[localId];
  //     if (from_vertexset->bool_map_[s] ) { 
  // 	for (int64_t ngh = sg->vertexArray[localId]; ngh < sg->vertexArray[localId+1]; ngh++) {
  // 	  WNode d = sg->edgeArray[ngh];
  // 	  if(  apply_func ( s , d.v, d.w ) ) { 
  // 	    next[d.v] = 1; 
  // 	  }
  // 	} //end of loop on in neighbors
  //     }
  //   } //end of outer for loop
  // }

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
	  if( apply_func ( s , d.v, d.w ) && CAS(&(g.flags_[d.v]), 0, 1)  ) { 
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
    SP_trackving_var_1 = writeMin( &SP[dst], (SP[src] + weight) ); 
    output2 = SP_trackving_var_1;
    return output2;
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadWeightedEdgesFromFile ( argv[(1) ]) ;
  edges.buildPushSegmentedGraphs("s1", atoi(argv[3]), true);
  std::cout<<"success"<<std::endl;
  SP = new int [ builtin_getVertices(edges) ];
  for ( int trail = (0) ; trail < (10) ; trail++ )
  {
    parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
      SP_generated_vector_op_apply_func_0()(i);
    };
    startTimer() ;
    int n = builtin_getVertices(edges) ;
    VertexSubset<int> *  frontier = new VertexSubset<int> ( builtin_getVertices(edges)  , (0) );
    builtin_addVertex(frontier, atoi(argv[2]) ) ;
    SP[atoi(argv[2]) ] = atoi(argv[2]) ;
    //int rounds = (0) ;
    while ( (builtin_getVertexSetSize(frontier) ) != ((0) ))
      {
	frontier = edgeset_apply_hybrid_denseforward_parallel_weighted_deduplicatied_from_vertexset_with_frontier(edges, frontier, updateEdge()); 
	// int result = 0;
	// for (NodeID n=0; n < builtin_getVertices(edges); n++) {
	//   result += SP[n];
	// }
	// std::cout << "result=" << result << std::endl;
	//rounds = (rounds + (1) );
	// if ((rounds) == (n))
	//   { 
	//     std::cout << "negative cycle"<< std::endl;
	//     break;
	//   } 
      }
    float elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
    //std::cout << "rounds"<< std::endl;
    //std::cout << rounds<< std::endl;
  }
};

