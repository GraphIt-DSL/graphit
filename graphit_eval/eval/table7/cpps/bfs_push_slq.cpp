#include <iostream> 
#include <vector>
#include "intrinsics.h"
#include <string>
Graph edges; 
std::vector< int  >  parent;
void parent_generated_vector_op_apply_func_0(NodeID v) 
{
  parent[v] =  -(1) ;
};
bool updateEdge(NodeID src, NodeID dst) 
{
  bool output2 ;
  bool parent_trackving_var_1 = (bool) 0;
  parent_trackving_var_1 = compare_and_swap ( parent[dst],  -(1) , src);
  output2 = parent_trackving_var_1;
  return output2;
};
bool toFilter(NodeID v) 
{
  bool output ;
  output = (parent[v]) == ( -(1) );
  return output;
};
int main(int argc, char * argv[] ) 
{
  edges = builtin_loadEdgesFromFile ( argv[(1) ]) ;
  parent = std::vector< int  >   ( builtin_getVertices(edges)  );
  
  for (int trail = 0; trail < 10; trail++){
 
    parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
      parent_generated_vector_op_apply_func_0(i);
    };
    startTimer() ;
    VertexSubset<int> *  frontier = new VertexSubset<int> ( builtin_getVertices(edges)  , (0) );
    builtin_addVertex(frontier, std::stoi(argv[2]) ) ;
    parent[std::stoi(argv[2]) ] = std::stoi(argv[2]) ;
    while ( (builtin_getVertexSetSize(frontier) ) != ((0) ))
      {
	frontier = edgeset_apply_push_parallel_sliding_queue_from_vertexset_with_frontier(edges, frontier, updateEdge); 
      }
    float elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
  }
}
