#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
Graph edges; 
double  * __restrict old_rank;
double  * __restrict new_rank;
int  * __restrict out_degree;
double  * __restrict contrib;
double  * __restrict error;
int  * __restrict generated_tmp_vector_2;
double damp; 
double beta_score; 
template <typename APPLY_FUNC > void edgeset_apply_pull_parallel(Graph & g , APPLY_FUNC apply_func) 
{ 
    int64_t numVertices = g.num_nodes(), numEdges = g.num_edges();
    parallel_for ( NodeID d=0; d < g.num_nodes(); d++) {
      for(NodeID s : g.in_neigh(d)){
	apply_func ( s , d );
      } //end of loop on in neighbors
    } //end of outer for loop
} //end of edgeset apply function 
struct error_generated_vector_op_apply_func_5
{
  void operator() (NodeID v) 
  {
    error[v] = ((float) 0) ;
  };
};
struct contrib_generated_vector_op_apply_func_4
{
  void operator() (NodeID v) 
  {
    contrib[v] = ((float) 0) ;
  };
};
struct generated_vector_op_apply_func_3
{
  void operator() (NodeID v) 
  {
    out_degree[v] = generated_tmp_vector_2[v];
  };
};
struct new_rank_generated_vector_op_apply_func_1
{
  void operator() (NodeID v) 
  {
    new_rank[v] = ((float) 0) ;
  };
};
struct old_rank_generated_vector_op_apply_func_0
{
  void operator() (NodeID v) 
  {
    old_rank[v] = (((float) 1)  / builtin_getVertices(edges) );
  };
};
struct computeContrib
{
  void operator() (NodeID v) 
  {
    contrib[v] = (old_rank[v] / out_degree[v]);
  };
};
struct updateEdge
{
  void operator() (NodeID src, NodeID dst) 
  {
    new_rank[dst] = (new_rank[dst] + contrib[src]);
  };
};
struct updateVertex
{
  void operator() (NodeID v) 
  {
    double old_score = old_rank[v];
    new_rank[v] = (beta_score + (damp * new_rank[v]));
    error[v] = fabs((new_rank[v] - old_rank[v])) ;
    old_rank[v] = new_rank[v];
    new_rank[v] = ((float) 0) ;
  };
};
struct printRank
{
  void operator() (NodeID v) 
  {
    std::cout << old_rank[v]<< std::endl;
  };
};
struct reset
{
  void operator() (NodeID v) 
  {
    old_rank[v] = (((float) 1)  / builtin_getVertices(edges) );
    new_rank[v] = ((float) 0) ;
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadEdgesFromFile ( argv[(1) ]) ;
  old_rank = new double [ builtin_getVertices(edges) ];
  new_rank = new double [ builtin_getVertices(edges) ];
  out_degree = new int [ builtin_getVertices(edges) ];
  contrib = new double [ builtin_getVertices(edges) ];
  error = new double [ builtin_getVertices(edges) ];
  generated_tmp_vector_2 = new int [ builtin_getVertices(edges) ];
  damp = ((float) 0.85) ;
  beta_score = ((((float) 1)  - damp) / builtin_getVertices(edges) );
  parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
    old_rank_generated_vector_op_apply_func_0()(i);
  };
  parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
    new_rank_generated_vector_op_apply_func_1()(i);
  };
  generated_tmp_vector_2 = builtin_getOutDegrees(edges) ;
  parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
    generated_vector_op_apply_func_3()(i);
  };
  parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
    contrib_generated_vector_op_apply_func_4()(i);
  };
  parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
    error_generated_vector_op_apply_func_5()(i);
  };
  for ( int trail = (0) ; trail < (10) ; trail++ )
  {
    parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
      reset()(i);
    };
    startTimer() ;
    for ( int i = (0) ; i < (20) ; i++ )
    {
      parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
        computeContrib()(i);
      };
      edgeset_apply_pull_parallel(edges, updateEdge()); 
      parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
        updateVertex()(i);
      };
    }
    double elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
  }
};

