#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
Graph edges; 
typedef struct struct_delta_out_degree { 
  double delta;
  int out_degree;
} struct_delta_out_degree;
double  * __restrict cur_rank;
double  * __restrict ngh_sum;
struct_delta_out_degree  * __restrict array_of_struct_delta_out_degree;
int  * __restrict generated_tmp_vector_3;
double damp; 
double beta_score; 
double epsilon2; 
double epsilon; 
template <typename APPLY_FUNC > void edgeset_apply_push_parallel_from_vertexset(Graph & g , VertexSubset<NodeID>* from_vertexset, APPLY_FUNC apply_func) 
{ 
    int64_t numVertices = g.num_nodes(), numEdges = g.num_edges();

    from_vertexset->toSparse();
    long m = from_vertexset->size();
  parallel_for (long i=0; i < m; i++) {
    NodeID s = from_vertexset->dense_vertex_set_[i];
    int j = 0;
    for(NodeID d : g.out_neigh(s)){
      apply_func ( s , d  );
    } //end of for loop on neighbors
  }
} //end of edgeset apply function 
struct generated_vector_op_apply_func_4
{
  void operator() (NodeID v) 
  {
    array_of_struct_delta_out_degree[v].out_degree  = generated_tmp_vector_3[v];
  };
};
struct delta_generated_vector_op_apply_func_2
{
  void operator() (NodeID v) 
  {
    array_of_struct_delta_out_degree[v].delta  = (((float) 1)  / builtin_getVertices(edges) );
  };
};
struct ngh_sum_generated_vector_op_apply_func_1
{
  void operator() (NodeID v) 
  {
    ngh_sum[v] = ((float) 0) ;
  };
};
struct cur_rank_generated_vector_op_apply_func_0
{
  void operator() (NodeID v) 
  {
    cur_rank[v] = (0) ;
  };
};
struct updateEdge
{
  void operator() (NodeID src, NodeID dst) 
  {
    writeAdd( &ngh_sum[dst], (array_of_struct_delta_out_degree[src].delta  / array_of_struct_delta_out_degree[src].out_degree ) ); 
  };
};
struct updateVertexFirstRound
{
  bool operator() (NodeID v) 
  {
    bool output ;
    array_of_struct_delta_out_degree[v].delta  = ((damp * ngh_sum[v]) + beta_score);
    cur_rank[v] += array_of_struct_delta_out_degree[v].delta ;
    array_of_struct_delta_out_degree[v].delta  = (array_of_struct_delta_out_degree[v].delta  - (((float) 1)  / builtin_getVertices(edges) ));
    output = (fabs(array_of_struct_delta_out_degree[v].delta ) ) > ((epsilon2 * cur_rank[v]));
    ngh_sum[v] = (0) ;
    return output;
  };
};
struct updateVertex
{
  bool operator() (NodeID v) 
  {
    bool output ;
    array_of_struct_delta_out_degree[v].delta  = (ngh_sum[v] * damp);
    cur_rank[v] += array_of_struct_delta_out_degree[v].delta ;
    ngh_sum[v] = (0) ;
    output = (fabs(array_of_struct_delta_out_degree[v].delta ) ) > ((epsilon2 * cur_rank[v]));
    return output;
  };
};
struct printRank
{
  void operator() (NodeID v) 
  {
    std::cout << cur_rank[v]<< std::endl;
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadEdgesFromFile ( argv[(1) ]) ;
  cur_rank = new double [ builtin_getVertices(edges) ];
  ngh_sum = new double [ builtin_getVertices(edges) ];
  array_of_struct_delta_out_degree = new struct_delta_out_degree [ builtin_getVertices(edges) ];
  generated_tmp_vector_3 = new int [ builtin_getVertices(edges) ];
  damp = ((float) 0.85) ;
  beta_score = ((((float) 1)  - damp) / builtin_getVertices(edges) );
  epsilon2 = ((float) 0.1) ;
  epsilon = ((float) 1e-07) ;
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    cur_rank_generated_vector_op_apply_func_0()(vertexsetapply_iter);
  };
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    ngh_sum_generated_vector_op_apply_func_1()(vertexsetapply_iter);
  };
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    delta_generated_vector_op_apply_func_2()(vertexsetapply_iter);
  };
  generated_tmp_vector_3 = builtin_getOutDegrees(edges) ;
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    generated_vector_op_apply_func_4()(vertexsetapply_iter);
  };
  int n = builtin_getVertices(edges) ;
  VertexSubset<int> *  frontier = new VertexSubset<int> ( builtin_getVertices(edges)  , n);
  startTimer() ;
  for ( int i = (1) ; i < (11) ; i++ )
  {
    edgeset_apply_push_parallel_from_vertexset(edges, frontier, updateEdge()); 
    if ((i) == ((1) ))
     { 
      auto ____graphit_tmp_out = new VertexSubset <NodeID> ( builtin_getVertices(edges)  , 0 );
bool * next5 = newA(bool, builtin_getVertices(edges) );
      parallel_for (int v = 0; v < builtin_getVertices(edges) ; v++) {
        next5[v] = 0;
if ( updateVertexFirstRound()( v ) )
          next5[v] = 1;
      } //end of loop
____graphit_tmp_out->num_vertices_ = sequence::sum( next5, builtin_getVertices(edges)  );
____graphit_tmp_out->bool_map_ = next5;

      frontier  = ____graphit_tmp_out; 
     } 
    else
     { 
      auto ____graphit_tmp_out = new VertexSubset <NodeID> ( builtin_getVertices(edges)  , 0 );
bool * next6 = newA(bool, builtin_getVertices(edges) );
      parallel_for (int v = 0; v < builtin_getVertices(edges) ; v++) {
        next6[v] = 0;
if ( updateVertex()( v ) )
          next6[v] = 1;
      } //end of loop
____graphit_tmp_out->num_vertices_ = sequence::sum( next6, builtin_getVertices(edges)  );
____graphit_tmp_out->bool_map_ = next6;

      frontier  = ____graphit_tmp_out; 

     } 
  }
  double elapsed_time = stopTimer() ;
  std::cout << "elapsed time: "<< std::endl;
  std::cout << elapsed_time<< std::endl;
};

