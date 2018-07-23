#include <iostream> 
#include <vector>
#include "intrinsics.h"
WGraph edges;
typedef double defined_type_0 [ 20]; 
defined_type_0 * __restrict  latent_vec;
typedef double defined_type_1 [ 20]; 
defined_type_1 * __restrict  error_vec;
double step; 
double lambda; 
int K; 
template <typename APPLY_FUNC > VertexSubset<NodeID>* edgeset_apply_pull_parallel_weighted_pull_edge_based_load_balance(WGraph & g , APPLY_FUNC apply_func) 
{ 
    long numVertices = g.num_nodes(), numEdges = g.num_edges();
    if (g.offsets_ == nullptr) g.SetUpOffsets(true);
  SGOffset * edge_in_index = g.offsets_;
    std::function<void(int,int,int)> recursive_lambda = 
    [&apply_func, &g,  &recursive_lambda, edge_in_index  ]
    (NodeID start, NodeID end, int grain_size){
         if ((start == end-1) || ((edge_in_index[end] - edge_in_index[start]) < grain_size)){
  for (NodeID d = start; d < end; d++){
    for(WNode s : g.in_neigh(d)){
      apply_func ( s.v , d, s.w );
    } //end of loop on in neighbors
   } //end of outer for loop
        } else { // end of if statement on grain size, recursive case next
                 cilk_spawn recursive_lambda(start, start + ((end-start) >> 1), grain_size);
                  recursive_lambda(start + ((end-start)>>1), end, grain_size);
        } 
    }; //end of lambda function
    recursive_lambda(0, numVertices, 4096);
    cilk_sync; 
  return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());
} //end of edgeset apply function 
struct updateEdge
{
  void operator() (NodeID src, NodeID dst, int rating) 
  {
    double estimate = (0) ;
    for ( int i = (0) ; i < K; i++ )
    {
      estimate += (latent_vec[src][i] * latent_vec[dst][i]);
    }
    double err = (rating - estimate);
    for ( int i = (0) ; i < K; i++ )
    {
      error_vec[dst][i] += (latent_vec[src][i] * err);
    }
  };
};
struct updateVertex
{
  void operator() (NodeID v) 
  {
    for ( int i = (0) ; i < K; i++ )
    {
      latent_vec[v][i] += (step * (( -lambda * latent_vec[v][i]) + error_vec[v][i]));
      error_vec[v][i] = (0) ;
    }
  };
};
struct initVertex
{
  void operator() (NodeID v) 
  {
    for ( int i = (0) ; i < K; i++ )
    {
      latent_vec[v][i] = ((float) 0.5) ;
      error_vec[v][i] = (0) ;
    }
  };
};
int main(int argc, char * argv[])
{
  edges = builtin_loadWeightedEdgesFromFile ( argv[(1) ]) ;
  latent_vec = new defined_type_0 [ builtin_getVertices(edges) ];
  error_vec = new defined_type_1 [ builtin_getVertices(edges) ];
  step = ((float) 3.5e-07) ;
  lambda = ((float) 0.001) ;
  K = (20) ;
  parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
    initVertex()(i);
  };
  startTimer() ;
  for ( int i = (0) ; i < (10) ; i++ )
  {
    edgeset_apply_pull_parallel_weighted_pull_edge_based_load_balance(edges, updateEdge()); 
    parallel_for (int i = 0; i < builtin_getVertices(edges) ; i++) {
      updateVertex()(i);
    };
  }
  double elapsed_time = stopTimer() ;
  std::cout << "elapsed time: "<< std::endl;
  std::cout << elapsed_time<< std::endl;
};

