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
double  **local_new_rank;
template <typename APPLY_FUNC > void edgeset_apply_pull_parallel(Graph & g , APPLY_FUNC apply_func) 
{ 
    int64_t numVertices = g.num_nodes(), numEdges = g.num_edges();
  parallel_for (int n = 0; n < numVertices; n++) {
    for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {
      local_new_rank[socketId][n] = new_rank[n];
    }
  }
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
    NodeID d = sg->graphId[localId];
    for (int64_t ngh = sg->vertexArray[localId]; ngh < sg->vertexArray[localId+1]; ngh++) {
      NodeID s = sg->edgeArray[ngh];
      apply_func ( s , d , socketId);
    } //end of loop on in neighbors
  } //end of outer for loop
} // end of per-socket parallel_for
    } // end of segment for loop
}// end of per-socket parallel region

  parallel_for (int n = 0; n < numVertices; n++) {
    for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {
      new_rank[n] += local_new_rank[socketId][n];
    }
  }
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
  void operator() (NodeID src, NodeID dst, int socketId) 
  {
    local_new_rank[socketId][dst] += contrib[src];
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
  edges.buildPullSegmentedGraphs("s1", atoi(argv[2]), true);
  old_rank = new double [ builtin_getVertices(edges) ];
  new_rank = new double [ builtin_getVertices(edges) ];
  out_degree = new int [ builtin_getVertices(edges) ];
  contrib = new double [ builtin_getVertices(edges) ];
  error = new double [ builtin_getVertices(edges) ];
  generated_tmp_vector_2 = new int [ builtin_getVertices(edges) ];
  damp = ((float) 0.85) ;
  beta_score = ((((float) 1)  - damp) / builtin_getVertices(edges) );
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    old_rank_generated_vector_op_apply_func_0()(vertexsetapply_iter);
  };
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    new_rank_generated_vector_op_apply_func_1()(vertexsetapply_iter);
  };
  generated_tmp_vector_2 = builtin_getOutDegrees(edges) ;
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    generated_vector_op_apply_func_3()(vertexsetapply_iter);
  };
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    contrib_generated_vector_op_apply_func_4()(vertexsetapply_iter);
  };
  parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
    error_generated_vector_op_apply_func_5()(vertexsetapply_iter);
  };
  local_new_rank = new double *[omp_get_num_places()];
  for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {
    local_new_rank[socketId] = (double *)numa_alloc_onnode(sizeof(double ) * builtin_getVertices(edges) , socketId);
    parallel_for (int n = 0; n < builtin_getVertices(edges) ; n++) {
      local_new_rank[socketId][n] = new_rank[n];
    }
  }
  omp_set_nested(1);
  for ( int trail = (0) ; trail < (10) ; trail++ )
  {
    startTimer() ;
    parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
      reset()(vertexsetapply_iter);
    };
    for ( int i = (0) ; i < (20) ; i++ )
    {
      parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
        computeContrib()(vertexsetapply_iter);
      };
      edgeset_apply_pull_parallel(edges, updateEdge()); 
      parallel_for (int vertexsetapply_iter = 0; vertexsetapply_iter < builtin_getVertices(edges) ; vertexsetapply_iter++) {
        updateVertex()(vertexsetapply_iter);
      };
    }
    double elapsed_time = stopTimer() ;
    std::cout << "elapsed time: "<< std::endl;
    std::cout << elapsed_time<< std::endl;
  }
  for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {
    numa_free(local_new_rank[socketId], sizeof(double ) * builtin_getVertices(edges) );
  }
};

