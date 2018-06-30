#include <iostream> 
#include <vector>
#include <algorithm>
#include "intrinsics.h"
WGraph edges;
typedef double defined_type_0 [ 20]; 
defined_type_0 * __restrict  latent_vec;
typedef double defined_type_1 [ 20]; 
defined_type_1 * __restrict  error_vec;
double step; 
double lambda; 
int K; 
template <typename APPLY_FUNC > void edgeset_apply_pull_parallel_weighted_pull_edge_based_load_balance(WGraph & g , APPLY_FUNC apply_func) 
{ 
    int64_t numVertices = g.num_nodes(), numEdges = g.num_edges();
    std::cout<<"|V|="<<numVertices<<std::endl;
    int64_t cnt=0;
    for (int segmentId = 0; segmentId < g.getNumSegments("s1"); segmentId++) {
      auto sg = g.getSegmentedGraph(std::string("s1"), segmentId);
      cnt+= sg->numVertices;
      std::cout<<"segment " << segmentId << " has " << sg->numVertices <<std::endl;
  if (g.offsets_ == nullptr) g.SetUpOffsets(true);
  SGOffset * edge_in_index = g.offsets_;
    std::function<void(int,int,int)> recursive_lambda = 
    [&apply_func, &g,  &recursive_lambda, edge_in_index, sg  ]
    (NodeID start, NodeID end, int grain_size){
         if ((start == end-1) || ((sg->vertexArray[end] - sg->vertexArray[start]) < grain_size)){
  for (NodeID localId = start; localId < end; localId++){
    NodeID d = sg->graphId[localId];
    for (int64_t ngh = sg->vertexArray[localId]; ngh < sg->vertexArray[localId+1]; ngh++) {
      WNode s = sg->edgeArray[ngh];
      apply_func ( s.v , d, s.w );
    } //end of loop on in neighbors
   } //end of outer for loop
        } else { // end of if statement on grain size, recursive case next
                 cilk_spawn recursive_lambda(start, start + ((end-start) >> 1), grain_size);
                  recursive_lambda(start + ((end-start)>>1), end, grain_size);
        } 
    }; //end of lambda function
    recursive_lambda(0, sg->numVertices, 4096);
    cilk_sync; 
    } // end of segment for loop
    std::cout<<"sum(|V_i|)="<<cnt<<std::endl;
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
  edges.buildPullSegmentedGraphs("s1", atoi(argv[2]));
  latent_vec = new defined_type_0 [ builtin_getVertices(edges) ];
  error_vec = new defined_type_1 [ builtin_getVertices(edges) ];
  step = ((float) 3.5e-07) ;
  lambda = ((float) 0.001) ;
  K = (20) ;
  for ( int trail = (0) ; trail < (10) ; trail++ )
  {
    startTimer() ;
    parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
      initVertex()(vertexsetapply_iter);
    };
    for ( int i = (0) ; i < (10) ; i++ )
    {
      edgeset_apply_pull_parallel_weighted_pull_edge_based_load_balance(edges, updateEdge()); 
      parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
        updateVertex()(vertexsetapply_iter);
      };
    }
    double elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
  }
};

