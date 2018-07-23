#include <iostream> 
#include <vector>
#include "intrinsics.h"

//#define VERIFY

WGraph edges;
int  * __restrict SP;

void SP_generated_vector_op_apply_func_0(NodeID v) 
{
  SP[v] = (2147483647) ;
};
bool updateEdge(NodeID src, NodeID dst, int weight) 
{
  bool output2 ;
  bool SP_trackving_var_1 = (bool) 0;
  SP_trackving_var_1 = writeMin( &SP[dst], (SP[src] + weight) ); 
  output2 = SP_trackving_var_1;
  return output2;
};
void printSP(NodeID v) 
{
  std::cout << SP[v]<< std::endl;
};
int main(int argc, char * argv[] ) 
{
  edges = builtin_loadWeightedEdgesFromFile (argv[1]) ;
  SP = new int [ builtin_getVertices(edges) ];

  for (int trail = 0; trail < 10; trail++){

    parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
      SP_generated_vector_op_apply_func_0(i);
    };
    startTimer() ;
    int n = builtin_getVertices(edges) ;
    VertexSubset<int> *  frontier = new VertexSubset<int> ( builtin_getVertices(edges)  , (0) );
    int sp = std::stoi(argv[2]);
    builtin_addVertex(frontier, sp ) ;
    SP[sp ] = (0) ;
    int rounds = (0) ;
    while ( (builtin_getVertexSetSize(frontier) ) != ((0) ))
      {
	frontier = edgeset_apply_push_parallel_sliding_queue_weighted_deduplicatied_from_vertexset_with_frontier(edges, frontier, updateEdge); 
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
  }

#ifdef VERIFY
  std::cout << "num_rounds: " << rounds << std::endl;
  int sum = 0;
  int count = 0;
  for (int i = 0; i < n; i++){
    if (SP[i] < 2147483647) {
      sum += SP[i];
      count++;
    }
  }
  std::cout << "SP sum: " << sum << std::endl;
  std::cout << "SP count: " << count << std::endl;
#endif

};

