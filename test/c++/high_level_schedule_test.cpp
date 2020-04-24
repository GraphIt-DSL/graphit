//
// Created by Yunming Zhang on 6/8/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/midend/midend.h>
#include <graphit/backend/backend.h>
#include <graphit/frontend/error.h>
#include <graphit/utils/exec_cmd.h>
#include <graphit/frontend/high_level_schedule.h>
#include <graphit/midend/mir.h>

using namespace std;
using namespace graphit;

class HighLevelScheduleTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        context_ = new graphit::FIRContext();
        errors_ = new std::vector<ParseError>();
        fe_ = new Frontend();
        mir_context_ = new graphit::MIRContext();

        const char* bfs_char = ("element Vertex end\n"
                                "element Edge end\n"
                                "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../../test/graphs/test.el\");\n"
                                "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                "const parent : vector{Vertex}(int) = -1;\n"
                                "func updateEdge(src : Vertex, dst : Vertex) "
                                "  parent[dst] = src; "
                                "end\n"
                                "func toFilter(v : Vertex) -> output : bool "
                                "  output = parent[v] == -1; "
                                "end\n"
                                "func main() "
                                "  var frontier : vertexset{Vertex} = new vertexset{Vertex}(0); "
                                "  frontier.addVertex(1); "
                                "  while (frontier.getVertexSetSize() != 0) "
                                "      #s1# frontier = edges.from(frontier).to(toFilter).applyModified(updateEdge, parent, true); "
                                "  end\n"
                                "  print \"finished running BFS\"; \n"
                                "end");


        const char*  pr_char = ("element Vertex end\n"
                                             "element Edge end\n"
                                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                             "const old_rank : vector{Vertex}(float) = 1.0;\n"
                                             "const new_rank : vector{Vertex}(float) = 0.0;\n"
                                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                                             "const error : vector{Vertex}(float) = 0.0;\n"
                                             "const damp : float = 0.85;\n"
                                             "const beta_score : float = (1.0 - damp) / vertices.size();\n"
                                             "func updateEdge(src : Vertex, dst : Vertex)\n"
                                             "    new_rank[dst] += old_rank[src] / out_degrees[src];\n"
                                             "end\n"
                                             "func updateVertex(v : Vertex)\n"
                                             "    new_rank[v] = beta_score + damp*(new_rank[v]);\n"
                                             "    error[v]    = fabs ( new_rank[v] - old_rank[v]);\n"
                                             "    old_rank[v] = new_rank[v];\n"
                                             "    new_rank[v] = 0.0;\n"
                                             "end\n"
                                             "func main()\n"
                                             "#l1# for i in 1:10\n"
                                             "   #s1# edges.apply(updateEdge);\n"
                                             "   #s2# vertices.apply(updateVertex);\n"
                                             "        print error.sum();"
                                             "    end\n"
                                             "end");

        const char*  export_pr_char = ("element Vertex end\n"
                                "element Edge end\n"
                                "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                                "const vertices : vertexset{Vertex};\n"
                                "const old_rank : vector{Vertex}(float);\n"
                                "const new_rank : vector{Vertex}(float);\n"
                                "const out_degrees : vector{Vertex}(int);\n"
                                "const damp : float = 0.85;\n"
                                "const beta_score : float;\n"
                                "func updateEdge(src : Vertex, dst : Vertex)\n"
                                "    new_rank[dst] += old_rank[src] / out_degrees[src];\n"
                                "end\n"
                                "func updateVertex(v : Vertex)\n"
                                "    new_rank[v] = beta_score + damp*(new_rank[v]);\n"
                                "    old_rank[v] = new_rank[v];\n"
                                "    new_rank[v] = 0.0;\n"
                                "end\n"
                                "func initVectors(v : Vertex) \n"
                                "   old_rank[v] = 1.0; \n"
                                "   new_rank[v] = 0.0; \n"
                                "end\n"
                                "export func export_func()\n"
                                "   edges = load (\"test.el\"); \n"
                                "   vertices = edges.getVertices(); \n"
                                "   old_rank = new vector{Vertex}(float)(); \n"
                                "   new_rank = new vector{Vertex}(float)(); \n"
                                "   out_degrees = edges.getOutDegrees(); \n"
                                "   beta_score = (1.0 - damp) / vertices.size(); \n"
                                "   vertices.apply(initVectors); \n"
                                "   for i in 1:10\n"
                                "       #s1# edges.apply(updateEdge);\n"
                                "       #s2# vertices.apply(updateVertex);\n"
                                "   end\n"
                                "end\n");

        const char * sssp_char =      "element Vertex end\n"
                                                         "element Edge end\n"
                                                         "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"../test/graphs/test.wel\");\n"
                                                         "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                                         "const SP : vector{Vertex}(int) = 2147483647; %should be INT_MAX \n"
                                                         "func updateEdge(src : Vertex, dst : Vertex, weight : int) -> output : bool\n"
                                                         "    SP[dst] min= (SP[src] + weight);\n"
                                                         "end\n"
                                                         "func main() \n"
                                                         "    var n : int = edges.getVertices();\n"
                                                         "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(0);\n"
                                                         "    frontier.addVertex(0); %add source vertex \n"
                                                         "    SP[0] = 0;\n"
                                                         "    var rounds : int = 0;\n"
                                                         "    while (frontier.getVertexSetSize() != 0)\n"
                                                         "         #s1# frontier = edges.from(frontier).applyModified(updateEdge, SP);\n"
                                                         "         rounds = rounds + 1;\n"
                                                         "         if rounds == n\n"
                                                         "             print \"negative cycle\";\n"
                                                         "          end\n"
                                                         "     end\n"
                                                         "end";

        const char * sssp_async_char =      "element Vertex end\n"
                "element Edge end\n"
                "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"../test/graphs/test.wel\");\n"
                "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                "const SP : vector{Vertex}(int) = 2147483647; %should be INT_MAX \n"
                "func updateEdge(src : Vertex, dst : Vertex, weight : int) -> output : bool\n"
                "    SP[dst] asyncMin= (SP[src] + weight);\n"
                "end\n"
                "func main() \n"
                "    var n : int = edges.getVertices();\n"
                "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(0);\n"
                "    frontier.addVertex(0); %add source vertex \n"
                "    SP[0] = 0;\n"
                "    var rounds : int = 0;\n"
                "    while (frontier.getVertexSetSize() != 0)\n"
                "         #s1# frontier = edges.from(frontier).applyModified(updateEdge, SP);\n"
                "         rounds = rounds + 1;\n"
                "         if rounds == n\n"
                "             print \"negative cycle\";\n"
                "          end\n"
                "     end\n"
                "end";


        const char* cc_char = ("element Vertex end\n"
                                                     "element Edge end\n"
                                                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/4.el\");\n"
                                                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                                     "const IDs : vector{Vertex}(int) = 1;\n"
                                                     "func updateEdge(src : Vertex, dst : Vertex)\n"
                                                     "    IDs[dst] min= IDs[src];\n"
                                                     "end\n"
                                                     "func init(v : Vertex)\n"
                                                     "     IDs[v] = v;\n"
                                                     "end\n"
                                                     "func printID(v : Vertex)\n"
                                                     "    print IDs[v];\n"
                                                     "end\n"
                                                     "func main()\n"
                                                     "    var n : int = edges.getVertices();\n"
                                                     "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(n);\n"
                                                     "    vertices.apply(init);\n"
                                                     "    while (frontier.getVertexSetSize() != 0)\n"
                                                     "        #s1# frontier = edges.from(frontier).applyModified(updateEdge, IDs);\n"
                                                     "    end\n"
                                                     "    vertices.apply(printID);\n"
                                                     "end");



        const char* cf_char = ( "element Vertex end\n"
                                                      "element Edge end\n"
                                                      "const edges : edgeset{Edge}(Vertex,Vertex, float) = load (argv[1]);\n"
                                                      "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                                      "const latent_vec : vector{Vertex}(vector[20](float));\n"
                                                      "const error_vec : vector{Vertex}(vector[20](float));\n"
                                                      "const step : float = 0.00000035;\n"
                                                      "const lambda : float = 0.001;\n"
                                                      "const K : int = 20;\n"
                                                      "func updateEdge (src : Vertex, dst : Vertex, rating : int)\n"
                                                      "    var estimate : float = 0;\n"
                                                      "    for i in 0:K\n"
                                                      "        estimate  += latent_vec[src][i] * latent_vec[dst][i];\n"
                                                      "    end\n"
                                                      "    var err : float = estimate - rating;\n"
                                                      "    for i in 0:K\n"
                                                      "        error_vec[dst][i] += latent_vec[src][i]*err;\n"
                                                      "    end\n"
                                                      "end\n"
                                                      "func updateVertex (v : Vertex)\n"
                                                      "     for i in 0:K\n"
                                                      "        latent_vec[v][i] += step*(-lambda*latent_vec[v][i] + error_vec[v][i]);\n"
                                                      "        error_vec[v][i] = 0;\n"
                                                      "     end\n"
                                                      "end\n"
                                                      "func initVertex (v : Vertex)\n"
                                                      "    for i in 0:K\n"
                                                      "        latent_vec[v][i] = 0.5;\n"
                                                      "        error_vec[v][i] = 0;\n"
                                                      "    end\n"
                                                      "end\n"
                                                      "func main()\n"
                                                      "    vertices.apply(initVertex);\n"
                                                      "    for i in 1:10\n"
                                                      "        #s1# edges.apply(updateEdge);\n"
                                                      "        vertices.apply(updateVertex);\n"
                                                      "    end\n"
                                                      "end"
        );


        const char* cc_pjump_char = ( "element Vertex end\n"
                                      "element Edge end\n"
                                      "\n"
                                      "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[1]);\n"
                                      "\n"
                                      "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                      "const IDs : vector{Vertex}(int) = 1;\n"
                                      "\n"
                                      "const update: vector[1](int);\n"
                                      "\n"
                                      "func updateEdge(src : Vertex, dst : Vertex)\n"
                                      "    IDs[dst] min= IDs[src];\n"
                                      "    %var src_id: Vertex = IDs[src];\n"
                                      "    %var dst_id: Vertex = IDs[dst];\n"
                                      "\n"
                                      "    %IDs[dst_id] min= IDs[src_id];\n"
                                      "    %IDs[src_id] min= IDs[dst_id];\n"
                                      "end\n"
                                      "\n"
                                      "func init(v : Vertex)\n"
                                      "     IDs[v] = v;\n"
                                      "end\n"
                                      "\n"
                                      "func pjump(v: Vertex) \n"
                                      "    var y: Vertex = IDs[v];\n"
                                      "    var x: Vertex = IDs[y];\n"
                                      "    if x != y\n"
                                      "        IDs[v] = x;\n"
                                      "        update[0] = 1;\n"
                                      "    end\n"
                                      "end\n"
                                      "\n"
                                      "func main()\n"
                                      "    var n : int = edges.getVertices();\n"
                                      "    for trail in 0:10\n"
                                      "        var frontier : vertexset{Vertex} = new vertexset{Vertex}(n);\n"
                                      "        startTimer();\n"
                                      "        vertices.apply(init);\n"
                                      "        while (frontier.getVertexSetSize() != 0)\n"
                                      "            #s1# var output: vertexset{Vertex} = edges.applyModified(updateEdge, IDs);\n"
                                      "\t    delete frontier;\n"
                                      "\t    frontier = output;\n"
                                      "            update[0] = 1;\n"
                                      "            while update[0] != 0\n"
                                      "\t\tupdate[0] = 0;\n"
                                      "\t\tvertices.apply(pjump);\n"
                                      "            end\n"
                                      "        end\n"
                                      "        var elapsed_time : float = stopTimer();\n"
                                      "\tdelete frontier;\n"
                                      "        print \"elapsed time: \";\n"
                                      "        print elapsed_time;\n"
                                      "    end\n"
                                      "end"
        );


        const char* prd_char =  ("element Vertex end\n"
                                                       "element Edge end\n"
                                                       "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[1]);\n"
                                                       "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                                       "const cur_rank : vector{Vertex}(float) = 1.0/vertices.size();\n"
                                                       "const ngh_sum : vector{Vertex}(float) = 0.0;\n"
                                                       "const delta : vector{Vertex}(float) = 1.0/vertices.size();\n"
                                                       "const out_degree : vector {Vertex}(int) = edges.getOutDegrees();\n"
                                                       "const error : vector{Vertex}(float) = 0.0;\n"
                                                       "const damp : float = 0.85;\n"
                                                       "const beta_score : float = (1.0 - damp) / vertices.size();\n"
                                                       "const epsilon2 : float = 0.01;\n"
                                                       "const epsilon : float = 0.0000001;\n"
                                                       "\n"
                                                       "func updateEdge(src : Vertex, dst : Vertex)\n"
                                                       "    ngh_sum[dst] += delta[src] /out_degree[src];\n"
                                                       "end\n"
                                                       "\n"
                                                       "func updateVertexFirstRound(v : Vertex) -> output : bool\n"
                                                       "    delta[v] = damp*(ngh_sum[v]) + beta_score;\n"
                                                       "    cur_rank[v] += delta[v];\n"
                                                       "    delta[v] = delta[v] - 1.0/vertices.size();\n"
                                                       "    output = (fabs(delta[v]) > epsilon2*cur_rank[v]);\n"
                                                       "    ngh_sum[v] = 0;"
                                                       "end\n"
                                                       "\n"
                                                       "func updateVertex(v : Vertex) -> output : bool\n"
                                                       "   delta[v] = ngh_sum[v]*damp;\n"
                                                       "   cur_rank[v]+= delta[v];\n"
                                                       "   output = fabs(delta[v]) > epsilon2*cur_rank[v];\n"
                                                       "   ngh_sum[v] = 0; "
                                                       "end\n"
                                                       "\n"
                                                       "func main()\n"
                                                       "    startTimer();\n"
                                                       "    var n : int = edges.getVertices();\n"
                                                       "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(n);\n"
                                                       "\n"
                                                       "    for i in 1:10\n"
                                                       "        #s1# edges.from(frontier).apply(updateEdge);\n"
                                                       "        if i == 1\n"
                                                       "            frontier = vertices.where(updateVertexFirstRound);\n"
                                                       "        else\n"
                                                       "            frontier = vertices.where(updateVertex);\n"
                                                       "        end\n"
                                                       "\n"
                                                       "    end\n"
                                                       "end");

        const char* prd_double_char =  ("element Vertex end\n"
                                                              "element Edge end\n"
                                                              "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[1]);\n"
                                                              "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                                              "const cur_rank : vector{Vertex}(double) = 1.0/vertices.size();\n"
                                                              "const ngh_sum : vector{Vertex}(double) = 0.0;\n"
                                                              "const delta : vector{Vertex}(double) = 1.0/vertices.size();\n"
                                                              "const out_degree : vector {Vertex}(int) = edges.getOutDegrees();\n"
                                                              "const error : vector{Vertex}(double) = 0.0;\n"
                                                              "const damp : double = 0.85;\n"
                                                              "const beta_score : double = (1.0 - damp) / vertices.size();\n"
                                                              "const epsilon2 : double = 0.01;\n"
                                                              "const epsilon : double = 0.0000001;\n"
                                                              "\n"
                                                              "func updateEdge(src : Vertex, dst : Vertex)\n"
                                                              "    ngh_sum[dst] += delta[src] /out_degree[src];\n"
                                                              "end\n"
                                                              "\n"
                                                              "func updateVertexFirstRound(v : Vertex) -> output : bool\n"
                                                              "    delta[v] = damp*(ngh_sum[v]) + beta_score;\n"
                                                              "    cur_rank[v] += delta[v];\n"
                                                              "    delta[v] = delta[v] - 1.0/vertices.size();\n"
                                                              "    output = (fabs(delta[v]) > epsilon2*cur_rank[v]);\n"
                                                              "    ngh_sum[v] = 0;"
                                                              "end\n"
                                                              "\n"
                                                              "func updateVertex(v : Vertex) -> output : bool\n"
                                                              "   delta[v] = ngh_sum[v]*damp;\n"
                                                              "   cur_rank[v]+= delta[v];\n"
                                                              "   output = fabs(delta[v]) > epsilon2*cur_rank[v];\n"
                                                              "   ngh_sum[v] = 0; "
                                                              "end\n"
                                                              "\n"
                                                              "func main()\n"
                                                              "    startTimer();\n"
                                                              "    var n : int = edges.getVertices();\n"
                                                              "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(n);\n"
                                                              "\n"
                                                              "    for i in 1:10\n"
                                                              "        #s1# edges.from(frontier).apply(updateEdge);\n"
                                                              "        if i == 1\n"
                                                              "            frontier = vertices.where(updateVertexFirstRound);\n"
                                                              "        else\n"
                                                              "            frontier = vertices.where(updateVertex);\n"
                                                              "        end\n"
                                                              "\n"
                                                              "    end\n"
                                                              "end");	
	
        const char*  pr_cc_char = ("element Vertex end\n"
                                             "element Edge end\n"
                                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                             "const IDs : vector{Vertex}(int) = 1;\n"				   
                                             "const old_rank : vector{Vertex}(float) = 1.0;\n"
                                             "const new_rank : vector{Vertex}(float) = 0.0;\n"
                                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                                             "const error : vector{Vertex}(float) = 0.0;\n"
                                             "const damp : float = 0.85;\n"
                                             "const beta_score : float = (1.0 - damp) / vertices.size();\n"
                                             "func updateEdgeCC(src : Vertex, dst : Vertex)\n"
                                             "    IDs[dst] min= IDs[src];\n"
                                             "end\n"
                                             "func init(v : Vertex)\n"
                                             "     IDs[v] = v;\n"
                                             "end\n"
                                             "func printID(v : Vertex)\n"
                                             "    print IDs[v];\n"
                                             "end\n"
                                             "func updateEdgePR(src : Vertex, dst : Vertex)\n"
                                             "    new_rank[dst] += old_rank[src] / out_degrees[src];\n"
                                             "end\n"
                                             "func updateVertex(v : Vertex)\n"
                                             "    new_rank[v] = beta_score + damp*(new_rank[v]);\n"
                                             "    error[v]    = fabs ( new_rank[v] - old_rank[v]);\n"
                                             "    old_rank[v] = new_rank[v];\n"
                                             "    new_rank[v] = 0.0;\n"
                                             "end\n"
                                             "func main()\n"
                                             "#l1# for i in 1:10\n"
                                             "   #s1# edges.apply(updateEdgePR);\n"
                                             "   #s2# vertices.apply(updateVertex);\n"
                                             "        print error.sum();"
				             "end\n"

				             "    var n : int = edges.getVertices();\n"
				             "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(n);\n"
				             "    vertices.apply(init);\n"
				             "    while (frontier.getVertexSetSize() != 0)\n"
				             "        #s1# frontier = edges.from(frontier).applyModified(updateEdgeCC, IDs);\n"
				             "    end\n"
				             "    vertices.apply(printID);\n"
                                             "end");


        const char* bc_char = ("element Vertex end\n"
                "element Edge end\n"
                "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/4.el\");\n"
                "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                "const num_paths : vector{Vertex}(double) = 0;\n"
                "const dependences : vector{Vertex}(double) = 0;\n"
                "const visited : vector{Vertex}(bool) = false;\n"
                "func forward_update(src : Vertex, dst : Vertex)\n"
                "    num_paths[dst] +=  num_paths[src];\n"
                "end\n"
                "func visited_vertex_filter(v : Vertex) -> output : bool\n"
                "    output = (visited[v] == false);\n"
                "end\n"
                "func mark_visited(v : Vertex)\n"
                "    visited[v] = true;\n"
                "end\n"
                "func mark_unvisited(v : Vertex)\n"
                "    visited[v] = false;\n"
                "end\n"
                "func backward_vertex_f(v : Vertex)\n"
                "    visited[v] = true;\n"
                "    dependences[v] += 1 / num_paths[v];\n"
                "end\n"
                "func backward_update(src : Vertex, dst : Vertex)\n"
                "    dependences[dst] += dependences[src];\n"
                "end\n"
                "func final_vertex_f(v : Vertex)\n"
                "    dependences[v] = (dependences[v] - 1 / num_paths[v]) * num_paths[v];\n"
                "end\n"
                "func main()\n"
                "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(0);\n"
                "    frontier.addVertex(8);\n"
                "    num_paths[8] = 1;\n"
                "    visited[8] = true;\n"
                "    var round : int = 0;\n"
                "    var frontier_list : list{vertexset{Vertex}} = new list{vertexset{Vertex}}();\n"
                "    % foward pass to propagate num_paths\n"
                "    while (frontier.getVertexSetSize() != 0)\n"
                "        round = round + 1;\n"
                "        #s1# var output : vertexset{Vertex} = edges.from(frontier).applyModified(forward_update, num_paths);\n"
                "        output.apply(mark_visited);\n"
                "        frontier_list.append(output);\n"
                "        frontier = output;\n"
                "    end\n"
                "    % transposing the edges\n"
                "    var transposed_edges : edgeset{Edge}(Vertex, Vertex) = edges.transpose();\n"
                "    % resetting the visited information for the backward pass\n"
                "    vertices.apply(mark_unvisited);\n"
                "    frontier.apply(backward_vertex_f);\n"
                "    frontier_list.pop();\n"
                "    % backward pass to accumulate the dependencies\n"
                "    while (round > 1)\n"
                "        round = round - 1;\n"
                "        #s2# transposed_edges.from(frontier).apply(backward_update);\n"
                "        frontier = frontier_list.pop();\n"
                "        frontier.apply(backward_vertex_f);\n"
                "    end\n"
                "    vertices.apply(final_vertex_f);\n"
                "end");

        const char* closeness_centrality_weighted_char = (
                "element Vertex end\n"
                "element Edge end\n"
                "const edges : edgeset{Edge}(Vertex, Vertex, int) = load (\"../../test/graphs/test_closeness_sssp.wel\");\n"
                "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                "const distance : vector{Vertex}(int) = 2147483647;\n"
                "func updateEdge(src : Vertex, dst : Vertex, weight : int) -> output : bool\n"
                        "    distance[dst] min= (distance[src] + weight);\n"
                 "end\n"
                "func toFilter(v : Vertex) -> output : bool\n"
                "     output = distance[v] == 2147483647;\n"
                "end\n"
                "func main()\n"
                "  var frontier : vertexset{Vertex} = new vertexset{Vertex}(0);\n"
                "  frontier.addVertex(1);\n"
                "  distance[1] = 0;\n"
                "  var n : int = edges.getVertices();\n"
                "  var rounds : int = 0;\n"
                "  while (frontier.getVertexSetSize() != 0)\n"
                "      frontier = edges.from(frontier).applyModified(updateEdge, distance);\n"
                "      rounds = rounds + 1;\n"
                "      if  rounds == n\n"
                "         print \"negative cycle\";\n"
                "         break;\n"
                "      end\n"
                "  end\n"
                "  var notConnected : vertexset{Vertex} = vertices.where(toFilter);\n"
                "  var amountNotConnected : int = notConnected.getVertexSetSize();\n"
                "  var sum: int = 0;\n"
                "  var numVerts : int = vertices.size();\n"
                "  for i in 0 : numVerts\n"
                "      sum += distance[i];\n"
                "  end\n"
                "  sum = sum + amountNotConnected;\n"
                "end");




        const char* delta_stepping_char = ("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"argv[1]\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const dist : vector{Vertex}(int) = 2147483647; %should be INT_MAX\n"
                             "const pq: priority_queue{Vertex}(int);"

                             "func updateEdge(src : Vertex, dst : Vertex, weight : int) \n"
                             "  var new_dist : int = dist[src] + weight; "
                             "  pq.updatePriorityMin(dst, dist[dst], new_dist); "
                             "end\n"
                             "func main() "
                             "  var start_vertex : int = atoi(argv[2]);"
                             " dist[start_vertex] = 0;"
                             "  pq = new priority_queue{Vertex}(int)(false, false, dist, 1, 2, false, start_vertex);"
                             "  while (pq.finished() == false) "
                             "    var frontier : vertexset{Vertex} = pq.dequeue_ready_set(); % dequeue_ready_set() \n"
                             "    #s1# edges.from(frontier).applyUpdatePriority(updateEdge);  \n"
                             "    delete frontier; "
                             "  end\n"
                             "end");

        const char* ppsp_char = ("element Vertex end\n"
                                           "element Edge end\n"
                                           "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"argv[1]\");\n"
                                           "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                           "const dist : vector{Vertex}(int) = 2147483647; %should be INT_MAX\n"
                                           "const pq: priority_queue{Vertex}(int);"

                                           "func updateEdge(src : Vertex, dst : Vertex, weight : int) \n"
                                           "  var new_dist : int = dist[src] + weight; "
                                           "  pq.updatePriorityMin(dst, dist[dst], new_dist); "
                                           "end\n"
                                           "func main() "
                                           "  var start_vertex : int = atoi(argv[2]);"
                                            "  var dst_vertex : int = atoi(argv[3]);"
                                            " dist[start_vertex] = 0;"
                                            "  pq = new priority_queue{Vertex}(int)(false, false, dist, 1, 2, false, start_vertex);"
                                           "  while (pq.finishedNode(dst_vertex) == false) "
                                           "    var frontier : vertexset{Vertex} = pq.dequeue_ready_set(); % dequeue_ready_set() \n"
                                           "    #s1# edges.from(frontier).applyUpdatePriority(updateEdge);  \n"
                                           "    delete frontier; "
                                           "  end\n"
                                           "end");

        const char* astar_char = ("element Vertex end\n"
                                 "element Edge end\n"
                                 "extern func load_coords(filename: string, num_nodes: int);\n"
                                 "extern func calculate_distance(source: Vertex, destination: Vertex) -> output: double;\n"
                                 "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"argv[1]\");\n"
                                 "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                 "const f_score : vector{Vertex}(int) = 2147483647; %should be INT_MAX\n"
                                 "const g_score : vector{Vertex}(int) = 2147483647; %should be INT_MAX\n"
                                 "const pq: priority_queue{Vertex}(int);"
                                 "const dst_vertex : Vertex;\n"
                                 "func updateEdge(src : Vertex, dst : Vertex, weight : int) \n"
                                 "  var new_f_score : int = f_score[src] + weight; "
                                 "  var changed : bool = writeMin(f_score[dst], new_f_score);"
                                 "  if changed \n"
                                 "    var new_g_score : int = max(new_f_score + calculate_distance(src, dst_vertex), g_score[src]);"
                                 "    pq.updatePriorityMin(dst, g_score[dst], new_g_score); "
                                 "  end\n"
                                 "end\n"
                                 "func main() "
                                 "  var start_vertex : int = atoi(argv[2]);"
                                 "  dst_vertex = atoi(argv[3]);"
                                 "  load_coords(argv[1]);"
                                 "  pq = new priority_queue{Vertex}(int)(false, false, g_score, 1, 2, false, start_vertex);"
                                 "  while (pq.finishedNode(dst_vertex) == false) "
                                 "    var frontier : vertexset{Vertex} = pq.dequeue_ready_set(); % dequeue_ready_set() \n"
                                 "    #s1# edges.from(frontier).applyUpdatePriority(updateEdge);  \n"
                                 "    delete frontier; "
                                 "  end\n"
                                 "end");

        const char* export_cf_char = ( "element Vertex end\n"
                                "element Edge end\n"
                                "const edges : edgeset{Edge}(Vertex,Vertex, float);\n"
                                "const vertices : vertexset{Vertex};\n"
                                "const latent_vec : vector{Vertex}(vector[20](float));\n"
                                "const error_vec : vector{Vertex}(vector[20](float));\n"
                                "const step : float = 0.00000035;\n"
                                "const lambda : float = 0.001;\n"
                                "const K : int = 20;\n"
                                "func updateEdge (src : Vertex, dst : Vertex, rating : int)\n"
                                "    var estimate : float = 0;\n"
                                "    for i in 0:K\n"
                                "        estimate  += latent_vec[src][i] * latent_vec[dst][i];\n"
                                "    end\n"
                                "    var err : float = estimate - rating;\n"
                                "    for i in 0:K\n"
                                "        error_vec[dst][i] += latent_vec[src][i]*err;\n"
                                "    end\n"
                                "end\n"
                                "func updateVertex (v : Vertex)\n"
                                "     for i in 0:K\n"
                                "        latent_vec[v][i] += step*(-lambda*latent_vec[v][i] + error_vec[v][i]);\n"
                                "        error_vec[v][i] = 0;\n"
                                "     end\n"
                                "end\n"
                                "func initVertex (v : Vertex)\n"
                                "    for i in 0:K\n"
                                "        latent_vec[v][i] = 0.5;\n"
                                "        error_vec[v][i] = 0;\n"
                                "    end\n"
                                "end\n"
                                "export func export_func(input_edges : edgeset{Edge}(Vertex,Vertex, float)) -> output : vector{Vertex}(vector[20](float))\n"
                                "    edges = input_edges;\n"
                                "    vertices  = edges.getVertices();\n"
                                "    latent_vec = new vector{Vertex}(vector[20](float))();\n"
                                "    error_vec = new vector{Vertex}(vector[20](float))();\n"
                                "    vertices.apply(initVertex);\n"
                                "    for i in 1:10\n"
                                "        #s1# edges.apply(updateEdge);\n"
                                "        vertices.apply(updateVertex);\n"
                                "    end\n"
                                "    output = latent_vec;\n"
                                "end"
        );


        const char* kcore_char =  ("element Vertex end\n"
                                 "element Edge end\n"
                                 "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                                 "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                 "const D: vector{Vertex}(int) = edges.getOutDegrees();\n"
                                 "const pq: priority_queue{Vertex}(int);\n"
                                 "func apply_f(src: Vertex, dst: Vertex)\n"
                                 "    var k: int = pq.get_current_priority();\n"
                                 "    pq.updatePrioritySum(dst, -1, k);\n"
                                 "end \n"
                                 "func main()\n"
                                 "    pq = new priority_queue{Vertex}(int)(false, false, D, 1, 2, true, -1);\n"
                                 "    var finished: int = 0; \n"
                                 "    while (finished != vertices.size()) \n"
                                 "        var frontier: vertexset{Vertex} = pq.dequeue_ready_set();\n"
                                 "        finished += frontier.size();\n"
                                   "        #s1# edges.from(frontier).applyUpdatePriority(apply_f);\n"
                                 "        delete frontier;\n"
                                 "    end\n"
                                 "    delete pq;\n"
                                 "end\n"
        );


        const char* kcore_uint_char =  ("element Vertex end\n"
                                   "element Edge end\n"
                                   "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                                   "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                   "const D: vector{Vertex}(uint) = edges.getOutDegreesUint();\n"
                                   "const pq: priority_queue{Vertex}(uint);\n"
                                   "func apply_f(src: Vertex, dst: Vertex)\n"
                                   "    var k: int = pq.get_current_priority();\n"
                                   "    pq.updatePrioritySum(dst, -1, k);\n"
                                   "end \n"
                                   "func main()\n"
                                   "    pq = new priority_queue{Vertex}(uint)(false, false, D, 1, 2, true, -1);\n"
                                   "    var finished: int = 0; \n"
                                   "    while (finished != vertices.size()) \n"
                                   "        var frontier: vertexset{Vertex} = pq.dequeue_ready_set();\n"
                                   "        finished += frontier.size();\n"
                                   "        #s1# edges.from(frontier).applyUpdatePriority(apply_f);\n"
                                   "        delete frontier;\n"
                                   "    end\n"
                                   "    delete pq;\n"
                                   "end\n"
        );

        const char* setcover_uint_char = ("element Vertex end\n"
                                     "element Edge end\n"
                                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                     "const degrees: vector{Vertex}(int) = edges.getOutDegrees();\n"
                                     "const D: vector{Vertex}(uint);\n"
                                     "const pq: priority_queue{Vertex}(uint);\n"
                                     "const epsilon: double = 0.01;\n"
                                     "const x: double = 1.0/log(1.0 + epsilon);\n"
                                     "func init_udf(v: Vertex) \n"
                                     "var deg: int = degrees[v];\n"
                                     "    if deg == 0\n"
                                     "        D[v] = 4294967295;\n"
                                     "    else\n"
                                     "        D[v] = floor(x*log(to_double(deg)));\n"
                                     "    end\n"
                                     "end\n"
                                     "extern func extern_function(active: vertexset{Vertex}) -> output: vertexset{Vertex};\n"
                                     "func main() \n"
                                     "    vertices.apply(init_udf);\n"
                                     "            pq = new priority_queue{Vertex}(uint)(false, false, D, 1, 2, false, -1);\n"
                                     "     while (1) \n"
                                     "        var frontier: vertexset{Vertex} = pq.get_min_bucket();\n"
                                     "        if frontier.is_null()\n"
                                     "            break;\n"
                                     "        end\n"
                                     "        frontier.applyUpdatePriorityExtern(extern_function);\n"
                                     "        delete frontier;\n"
                                     "    end\n"
                                     "end\n");

        const char* unordered_kcore_char = ("element Vertex end\n"
                                            "element Edge end\n"
                                            "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                                            "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                            "const Degrees : vector {Vertex}(int) = edges.getOutDegrees();\n"
                                            "const coreNumbers : vector {Vertex}(int) = 0;\n"
                                            "const k : int;\n"
                                            "\n"
                                            "func updateDegrees(src : Vertex, dst : Vertex)\n"
                                            "    Degrees[dst] += -1;\n"
                                            "end\n"
                                            "\n"
                                            "func filter1(v: Vertex) -> output : bool \n"
                                            "     if Degrees[v] < k\n"
                                            "       coreNumbers[v] = k-1;\n"
                                            "       Degrees[v] = 0;\n"
                                            "       output = true;\n"
                                            "     else\n"
                                            "       output = false;\n"
                                            "     end\n"
                                            "end\n"
                                            "\n"
                                            "func filter2(v: Vertex) -> output : bool output = (Degrees[v] >= k); end\n"
                                            "\n"
                                            "func main()\n"
                                            "    startTimer();\n"
                                            "    var n : int = edges.getVertices();\n"
                                            "    var largestCore : int = -1;\n"
                                            "    var frontier : vertexset{Vertex} = new vertexset{Vertex}(n);\n"
                                            "    for iter in 1:n;\n"
                                            "        k = iter;\n"
                                            "        while(1) \n"
                                            "          var toRemove : vertexset{Vertex} = frontier.where(filter1);\n"
                                            "          var remaining : vertexset{Vertex}  = frontier.where(filter2); %remaining vertices\n"
                                            "          delete frontier; \n"
                                            "          frontier = remaining; \n"
                                            "          if (0 == toRemove.getVertexSetSize())\n"
                                            "             break;\t\n"
                                            "          else\n"
                                            "             #s1# edges.from(toRemove).apply(updateDegrees);\n"
                                            "             delete toRemove; \n"
                                            "          end \n"
                                            "        end\n"
                                            "        if (0 == frontier.getVertexSetSize())\n"
                                            "          largestCore = k-1;\n"
                                            "          break;\n"
                                            "        end\n"
                                            "    end\n"
                                            "    var elapsed_time : float = stopTimer();\n"
                                            "    print \"elapsed time: \";\n"
                                            "    print elapsed_time;\n"
                                            "end");

        const char* simple_intersection = ("element Vertex end\n"
                                           "element Edge end\n"
                                           "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                                           "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                                           "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                                           "func main() "
                                           "#s1# const inter: uint_64 = intersection(vertices1, vertices2, 0, 0);\n"
                                           "end\n");

        const char* simple_intersection_opt = ("element Vertex end\n"
                                           "element Edge end\n"
                                           "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                                           "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                                           "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                                           "func main() "
                                           "#s1# const inter: uint_64 = intersection(vertices1, vertices2, 0, 0, 5);\n"
                                           "end\n");

        const char* simple_intersect_neigh_opt = ("element Vertex end\n"
                                               "element Edge end\n"
                                               "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                                               "const src : int = 0;\n"
                                               "const dest : int = 1;\n"
                                               "func main() "
                                               "#s1# const inter: uint_64 = intersectNeighbor(edges, src, dest);\n"
                                               "end\n");

        const char* bc_functor = ("element Vertex end\n"
                                 "element Edge end\n"
                                 "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                                 "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                                 "const simpleArray: vector{Vertex}(int) = 0;\n"
                                 "const visited: vector{Vertex}(bool) = false;"
                                 "func update_edge[a: vector{Vertex}(int)](src: Vertex, dst: Vertex)\n"
                                 "    a[src] += a[dst];"
                                 "end\n"
                                 "func visited_filter(v : Vertex) -> output : bool\n"
                                 "     output = (visited[v] == false);\n"
                                 "end\n"
                                 "func main()\n"
                                 "var frontier : vertexset{Vertex} = new vertexset{Vertex}(0);\n"
                                 "frontier.addVertex(3)\n;"
                                 "    var local_array: vector{Vertex}(int) = 0;"
                                 "    #s1# edges.from(frontier).to(visited_filter).applyModified(update_edge[local_array], local_array);\n"
                                 "end\n");

        bfs_str_ =  string (bfs_char);
        pr_str_ = string(pr_char);
        sssp_str_ = string  (sssp_char);
        sssp_async_str_ = string (sssp_async_char);
        cf_str_ = string  (cf_char);
        cc_str_ = string  (cc_char);
        cc_pjump_str_ = string  (cc_pjump_char);
        prd_str_ = string  (prd_char);
        prd_double_str_ = string  (prd_double_char);
        pr_cc_str_ = string(pr_cc_char);
        bc_str_ = string(bc_char);
        closeness_centrality_weighted_str_ = string(closeness_centrality_weighted_char);
        delta_stepping_str_ = string(delta_stepping_char);
        ppsp_str_ = string(ppsp_char);
        astar_str_ = string(astar_char);
        export_pr_str_ = string(export_pr_char);
        export_cf_str_ = string(export_cf_char);
        kcore_str_ = string(kcore_char);
        kcore_uint_str_ = string(kcore_uint_char);
        unordered_kcore_str_ = string (unordered_kcore_char);
        setcover_uint_str_ = string(setcover_uint_char);
        simple_intersection_str_ = string(simple_intersection);
        simple_intersection_opt_str_ = string(simple_intersection_opt);
        simple_intersect_neigh_opt_str_ = string(simple_intersect_neigh_opt);
        bc_functor_str_ = string(bc_functor);
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

        //prints out the MIR, just a hack for now
        //std::cout << "mir: " << std::endl;
        //std::cout << *(mir_context->getStatements().front());
        //std::cout << std::endl;

    }

    int basicTest(std::istream &is) {
        fe_->parseStream(is, context_, errors_);
        graphit::Midend *me = new graphit::Midend(context_);

        std::cout << "fir: " << std::endl;
        std::cout << *(context_->getProgram());
        std::cout << std::endl;

        me->emitMIR(mir_context_);
        graphit::Backend *be = new graphit::Backend(mir_context_);
        return be->emitCPP();
    }

/**
 * This test assumes that the fir_context is constructed in the specific test code
 * @return
 */
    int basicCompileTestWithContext() {
        graphit::Midend *me = new graphit::Midend(context_);
        me->emitMIR(mir_context_);
        graphit::Backend *be = new graphit::Backend(mir_context_);
        return be->emitCPP();
    }


    int basicTestWithSchedule(
            fir::high_level_schedule::ProgramScheduleNode::Ptr program) {

        graphit::Midend *me = new graphit::Midend(context_, program->getSchedule());
        std::cout << "fir: " << std::endl;
        std::cout << *(context_->getProgram());
        std::cout << std::endl;

        me->emitMIR(mir_context_);
        graphit::Backend *be = new graphit::Backend(mir_context_);
        return be->emitCPP();
    }

    std::vector<ParseError> *errors_;
    graphit::FIRContext *context_;
    Frontend *fe_;
    graphit::MIRContext *mir_context_;

    string bfs_str_;
    string pr_str_;
    string sssp_str_;
    string sssp_async_str_;
    string cf_str_;
    string cc_str_;
    string cc_pjump_str_;
    string prd_str_;
    string prd_double_str_;
    string pr_cc_str_;
    string bc_str_;
    string closeness_centrality_weighted_str_;
    string delta_stepping_str_;
    string ppsp_str_;
    string astar_str_;
    string export_pr_str_;
    string export_cf_str_;
    string kcore_str_;
    string kcore_uint_str_;
    string setcover_uint_str_;
    string unordered_kcore_str_;
    string simple_intersection_str_;
    string simple_intersection_opt_str_;
    string simple_intersect_neigh_opt_str_;
    string bc_functor_str_;
};

TEST_F(HighLevelScheduleTest, SimpleStructHighLevelSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vector_b : vector{Vertex}(float) = 0.0;\n"
    );

    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->fuseFields("vector_a", "vector_b");

    EXPECT_EQ (0, basicTestWithSchedule(program));

}

TEST_F(HighLevelScheduleTest, FuseMoreThanTwoFieldVectors) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vector_b : vector{Vertex}(float) = 0.0;\n"
                             "const vector_c : vector{Vertex}(float) = 0.0;\n"
    );

    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->fuseFields({"vector_a", "vector_b", "vector_c"});

    EXPECT_EQ (0, basicTestWithSchedule(program));

}

TEST_F(HighLevelScheduleTest, EdgeSetGetOutDegreesFuseStruct) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                             "const old_rank : vector{Vertex}(int) = 0;\n"
                             "func main()  end");

    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->fuseFields("out_degrees", "old_rank");

    EXPECT_EQ (0, basicTestWithSchedule(program));
}

/**
 * A test case that tries to break the 10 iters loop into a 2 iters and a 8 iters loop
 */
TEST_F(HighLevelScheduleTest, SimpleLoopIndexSplit) {
    istringstream is("func main() "
                             "for i in 1:10; print i; end "
                             "end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->splitForLoop("l1", "l2", "l3", 2, 8);


    //generate c++ code successfully
    EXPECT_EQ (0, basicCompileTestWithContext());

    //expects two loops in the main function decl
    EXPECT_EQ (2, main_func_decl->body->stmts.size());

}

/**
 * A test case that tries to break the 10 iters loop into a 2 iters and a 8 iters loop
 */
TEST_F(HighLevelScheduleTest, SimpleLoopIndexSplitWithLabelParsing) {
    istringstream is("func main() "
                             "# l1 # for i in 1:10; print i; end "
                             "end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->splitForLoop("l1", "l2", "l3", 2, 8);


    //generate c++ code successfully
    EXPECT_EQ (0, basicCompileTestWithContext());

    //expects two loops in the main function decl
    EXPECT_EQ (2, main_func_decl->body->stmts.size());

}

TEST_F(HighLevelScheduleTest, HighLevelApplyFunctionFusion) {

    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "func srcAddOne(src : Vertex, dst : Vertex) "
                             "vector_a[src] = vector_a[src] + 1; end\n"
                             "func srcAddTwo(src : Vertex, dst : Vertex) "
                             "vector_a[src] = vector_a[src] + 2; end\n"
                             "func main() "
                             "  edges.apply(srcAddOne); "
                             "  edges.apply(srcAddTwo); "
                             "end");

    fe_->parseStream(is, context_, errors_);

    //set up the labels
    fir::FuncDecl::Ptr main_func = fir::to<fir::FuncDecl>(context_->getProgram()->elems[7]);
    fir::ExprStmt::Ptr first_apply = fir::to<fir::ExprStmt>(main_func->body->stmts[0]);
    fir::ExprStmt::Ptr second_apply = fir::to<fir::ExprStmt>(main_func->body->stmts[1]);
    first_apply->stmt_label = "l1";
    second_apply->stmt_label = "l2";

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    //program_schedule_node = program_schedule_node->fuseApplyFunctions("l1", "l2", "l3","fused_func");
    program_schedule_node = program_schedule_node->fuseApplyFunctions("l1", "l2","fused_func");

    main_func = fir::to<fir::FuncDecl>(context_->getProgram()->elems[7]);
    first_apply = fir::to<fir::ExprStmt>(main_func->body->stmts[0]);

    // Expects that the program still compiles
    EXPECT_EQ (0,  basicCompileTestWithContext());

    // Expects one more fused function declarations now
    EXPECT_EQ (9,  context_->getProgram()->elems.size());
}


TEST_F(HighLevelScheduleTest, SimpleLoopAndKernelFusion) {

    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "func srcAddOne(src : Vertex, dst : Vertex) "
                             "vector_a[src] += 1; end\n"
                             "func srcAddTwo(src : Vertex, dst : Vertex) "
                             "vector_a[src] += 2; end\n"
                             "func main() "
                             "  #l1# for i in 1:10 "
                             "      #s1# edges.apply(srcAddOne); "
                             "  end "
                             "  #l2# for i in 1:10 "
                             "      #s1# edges.apply(srcAddTwo); "
                             "  end "
                             "end");

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program_schedule_node = program_schedule_node->fuseForLoop("l1", "l2", "l3");
    program_schedule_node = program_schedule_node->fuseApplyFunctions("l3:l1:s1", "l3:l2:s1", "fused_func");
    program_schedule_node->configApplyParallelization("l3:l1:s1", "dynamic-vertex-parallel");
    // Expects that the program still compiles
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

}


TEST_F(HighLevelScheduleTest, SimpleLabelForVarDecl) {

    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "func srcAddOne(src : Vertex, dst : Vertex) "
                             "vector_a[src] += 1; end\n"
                             "func srcAddTwo(src : Vertex, dst : Vertex) "
                             "vector_a[src] += 2; end\n"
                             "func main() "
                             "    #s1# var output : vertexset{Vertex} = edges.applyModified(srcAddOne, vector_a); "
                             "end");

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program_schedule_node = program_schedule_node->configApplyDirection("s1", "SparsePush");
    program_schedule_node->configApplyParallelization("s1", "dynamic-vertex-parallel");
    // Expects that the program still compiles
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::VarDecl::Ptr var_decl = mir::to<mir::VarDecl>((*(main_func_decl->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(var_decl->initVal));

}

TEST_F(HighLevelScheduleTest, SimpleLabelForVarDeclWithDifferentGrainSize) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const vector_a : vector{Vertex}(float) = 0.0;\n"
                     "func srcAddOne(src : Vertex, dst : Vertex) "
                     "vector_a[src] += 1; end\n"
                     "func srcAddTwo(src : Vertex, dst : Vertex) "
                     "vector_a[src] += 2; end\n"
                     "func main() "
                     "    #s1# edges.apply(srcAddOne); "
                     "end");

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program_schedule_node = program_schedule_node->configApplyDirection("s1", "SparsePush");
    program_schedule_node->configApplyParallelization("s1", "dynamic-vertex-parallel", 1);
    // Expects that the program still compiles
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

}

TEST_F(HighLevelScheduleTest, SimpleIntersectionHiroshi) {
    istringstream is(simple_intersection_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configIntersection("s1", "HiroshiIntersection");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, SimpleIntersectionMultiskip) {
    istringstream is(simple_intersection_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configIntersection("s1", "MultiskipIntersection");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, SimpleIntersectionCombined) {
    istringstream is(simple_intersection_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configIntersection("s1", "CombinedIntersection");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, SimpleIntersectionBinary) {
    istringstream is(simple_intersection_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configIntersection("s1", "BinarySearchIntersection");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, SimpleIntersectionWithOptional) {
    istringstream is(simple_intersection_opt_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configIntersection("s1", "HiroshiIntersection");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, SimpleIntersectionWithDifferentScheduler) {
    istringstream is(simple_intersection_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configIntersection("s2", "HiroshiIntersection");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, SimpleIntersectNeigh) {
    istringstream is(simple_intersect_neigh_opt_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configIntersection("s1", "HiroshiIntersection");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, BCFunctorTest) {
    istringstream is(bc_functor_str_);

    fe_->parseStream(is, context_, errors_);

    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program = program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, BFSPushSerialSchedule) {
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush");
    program->setApply("s1", "disable_deduplication");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(assign_stmt->expr));
}


TEST_F(HighLevelScheduleTest, BFSPushParallelSchedule) {
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel")
            ->setApply("s1", "disable_deduplication");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(assign_stmt->expr));
}


TEST_F(HighLevelScheduleTest, BFSPushSlidingQueueSchedule) {
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush");
    program->setApply("s1", "sliding_queue")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, BFSPullSchedule) {
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "DensePull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(assign_stmt->expr));
}


TEST_F(HighLevelScheduleTest, BFSPullEdgeAwareParallelSchedule) {
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "DensePull");
    program->setApply("s1", "disable_deduplication");
    program->configApplyParallelization("s1", "edge-aware-dynamic-vertex-parallel",1024, "DensePull");


    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, BFSHybridDenseSchedule) {
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->setApply("s1", "disable_deduplication");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::HybridDenseEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, CCNoSchedule) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, CCPJUMPNoSchedule) {
    istringstream is (cc_pjump_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, CCHybridDenseSchedule) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[3]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::HybridDenseEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, CCHybridDenseTwoSegments) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->configApplyNumSSG("s1", "fixed-vertex-count",  2, "DensePull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[3]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::HybridDenseEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, CCHybridDenseBitvectorFrontierSchedule) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->configApplyDenseVertexSet("s1", "bitvector", "src-vertexset", "DensePull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[3]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::HybridDenseEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, CCHybridDenseBitvectorFrontierScheduleNewAPI) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush-DensePull")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel")
            ->configApplyDenseVertexSet("s1", "bitvector", "src-vertexset", "DensePull")
            ->configApplyNumSSG("s1", "fixed-vertex-count", 4, "DensePull")
            ->configApplyNUMA("s1", "serial", "DensePull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[3]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::HybridDenseEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, CCPullSchedule) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[3]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(assign_stmt->expr));
}


TEST_F(HighLevelScheduleTest, CCPushSchedule) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->configApplyDirection("s1", "SparsePush")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[3]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, PRNestedSchedule) {
    istringstream is (pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->fuseFields("old_rank", "out_degrees")->splitForLoop("l1", "l2", "l3", 2, 8);
    program->setApply("l2:s1", "push")->setApply("l3:s1", "pull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    // 2 for the splitted for loops
    EXPECT_EQ(2, main_func_decl->body->stmts->size());

    // the first apply should be push
    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(expr_stmt->expr));

    // the second apply should be pull
    for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[1]);
    expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));

}


TEST_F(HighLevelScheduleTest, PRPullParallel) {
    istringstream is (pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->configApplyDirection("l1:s1", "DensePull")
            ->configApplyParallelization("l1:s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    // the first apply should be push
    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));

}

TEST_F(HighLevelScheduleTest, PRPullParallelTwoSegments) {
    istringstream is (pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // Set the pull parameter to 2 segments
    program->configApplyDirection("l1:s1", "DensePull")->configApplyParallelization("l1:s1", "dynamic-vertex-parallel");
    program->configApplyNumSSG("l1:s1", "fixed-vertex-count",  2);
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));

}


TEST_F(HighLevelScheduleTest, PRPullParallelRuntimeSegmentArgs) {
    istringstream is (pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // Set the pull parameter to 2 segments
    program->configApplyDirection("l1:s1", "DensePull")->configApplyParallelization("l1:s1", "dynamic-vertex-parallel");
    program->configApplyNumSSG("l1:s1", "fixed-vertex-count",  "argv[1]");
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));

}





TEST_F(HighLevelScheduleTest, PRPullParallelNumaAware) {
    istringstream is (pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // Set the pull parameter to 2 segments
    program->configApplyDirection("l1:s1", "DensePull")->configApplyParallelization("l1:s1", "dynamic-vertex-parallel");
    program->configApplyNumSSG("l1:s1", "fixed-vertex-count",  2, "DensePull");
    program->configApplyNUMA("l1:s1", "static-parallel", "DensePull");
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));

}

TEST_F(HighLevelScheduleTest, PRPullVertexsetParallel) {
    istringstream is (pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->configApplyDirection("l1:s1", "DensePull")->configApplyParallelization("l1:s1", "dynamic-vertex-parallel");
    program->configApplyParallelization("l1:s2", "dynamic-vertex-parallel");

    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    // the first apply should be push
    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));

}


TEST_F(HighLevelScheduleTest, PRPushParallel) {
    istringstream is (pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->configApplyDirection("l1:s1", "SparsePush")->configApplyParallelization("l1:s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    // the first apply should be push
    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(expr_stmt->expr));

}


TEST_F(HighLevelScheduleTest, SimpleHighLevelLoopFusion) {
    istringstream is("func main() "
                             "for i in 1:10; print i; end "
                             "for i in 1:10; print i+1; end "
                             "end");

    fe_->parseStream(is, context_, errors_);
    //attach the labels "l1" and "l2" to the for loop statements
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::ForStmt::Ptr l2_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);
    l1_loop->stmt_label = "l1";
    l2_loop->stmt_label = "l2";

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node = program_schedule_node->fuseForLoop("l1", "l2", "l3");

    main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l3_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);

    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    //generate c++ code successfully
    EXPECT_EQ (0,  basicCompileTestWithContext());

    //ony l3 loop statement left
    EXPECT_EQ (1,  main_func_decl->body->stmts.size());

    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_stmt =
            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop, "l3");

    //the body of l3 loop should consists of 2 statements
    EXPECT_EQ(2, schedule_for_stmt->getBody()->getNumStmts());

    auto fir_stmt_blk = schedule_for_stmt->getBody()->emitFIRNode();
    //expects both statements of the l3 loop body to be print statements
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[0]));
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[1]));

    auto fir_name_node = fir::to<fir::NameNode>(fir_stmt_blk->stmts[0]);
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_name_node->body->stmts[0]));

}

TEST_F(HighLevelScheduleTest, HighLevelLoopFusionPrologueEpilogue1) {
    istringstream is("func main() "
                             "for i in 1:10; print i; end "
                             "for i in 3:7; print i+1; end "
                             "end");

    fe_->parseStream(is, context_, errors_);
    //attach the labels "l1" and "l2" to the for loop statements
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::ForStmt::Ptr l2_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);
    l1_loop->stmt_label = "l1";
    l2_loop->stmt_label = "l2";

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node = program_schedule_node->fuseForLoop("l1", "l2", "l3");

    main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l3_loop_prologue = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::ForStmt::Ptr l3_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);
    fir::ForStmt::Ptr l3_loop_epilogue = fir::to<fir::ForStmt>(main_func_decl->body->stmts[2]);

    //generate c++ code successfully
    EXPECT_EQ (0,  basicCompileTestWithContext());

    //Check that 3 loops were generated: the prologue, the l3 loop (the main loop), and the epilogue loop
    EXPECT_EQ (3,  main_func_decl->body->stmts.size());

    // Check the correctness of the L3 loop generated.
    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_stmt =
            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop, "l3");

    //the body of l3 loop should consists of 2 statements
    EXPECT_EQ(2, schedule_for_stmt->getBody()->getNumStmts());

    auto fir_stmt_blk = schedule_for_stmt->getBody()->emitFIRNode();
    //expects both statements of the l3 loop body to be print statements
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[0]));
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[1]));

    auto fir_name_node = fir::to<fir::NameNode>(fir_stmt_blk->stmts[0]);
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_name_node->body->stmts[0]));
    auto fir_name_node2 = fir::to<fir::NameNode>(fir_stmt_blk->stmts[1]);
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_name_node2->body->stmts[0]));

    // Check the correctness of the L3 prologue loop generated.
    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_prologue_stmt =
                std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop_prologue, "l3_prologue");

    //the body of l3_prologue loop should consists of 1 statement
    EXPECT_EQ(1, schedule_for_prologue_stmt->getBody()->getNumStmts());

    auto fir_stmt_prologue_blk = schedule_for_prologue_stmt->getBody()->emitFIRNode();
    //expects the statement of the l3 prologue loop body to be a print statement
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_prologue_blk->stmts[0]));

    // Check the correctness of the L3 epilogue loop generated.
    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_epilogue_stmt =
                std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop_epilogue, "l3_epilogue");

    //the body of l3_epilogue loop should consists of 1 statement
    EXPECT_EQ(1, schedule_for_epilogue_stmt->getBody()->getNumStmts());

    auto fir_stmt_epilogue_blk = schedule_for_epilogue_stmt->getBody()->emitFIRNode();
    //expects the statement of the l3 epilogue loop body to be a print statement
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_epilogue_blk->stmts[0]));
    EXPECT_EQ("l1", fir::to<fir::NameNode>(fir_stmt_epilogue_blk->stmts[0])->stmt_label);

}


TEST_F(HighLevelScheduleTest, HighLevelLoopFusionPrologueEpilogue2) {
    istringstream is("func main() "
                             "for i in 1:10; print i; end "
                             "for i in 1:7; print i+1; end "
                             "end");

    fe_->parseStream(is, context_, errors_);
    //attach the labels "l1" and "l2" to the for loop statements
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::ForStmt::Ptr l2_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);
    l1_loop->stmt_label = "l1";
    l2_loop->stmt_label = "l2";

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node = program_schedule_node->fuseForLoop("l1", "l2", "l3");

    main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l3_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
        fir::ForStmt::Ptr l3_loop_epilogue = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);

    //generate c++ code successfully
    EXPECT_EQ (0,  basicCompileTestWithContext());

    //Check that 2 loops were generated: the l3 loop (the main loop), and the epilogue loop
    EXPECT_EQ (2,  main_func_decl->body->stmts.size());

    // Check the correctness of the L3 loop generated.
    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_stmt =
            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop, "l3");

    //the body of l3 loop should consists of 2 statements
    EXPECT_EQ(2, schedule_for_stmt->getBody()->getNumStmts());

    auto fir_stmt_blk = schedule_for_stmt->getBody()->emitFIRNode();
    //expects both statements of the l3 loop body to be print statements
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[0]));
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[1]));
    auto fir_name_node = fir::to<fir::NameNode>(fir_stmt_blk->stmts[0]);
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_name_node->body->stmts[0]));
    auto fir_name_node2 = fir::to<fir::NameNode>(fir_stmt_blk->stmts[1]);
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_name_node2->body->stmts[0]));
    EXPECT_EQ("l1", fir_name_node->stmt_label);
    EXPECT_EQ("l2", fir_name_node2->stmt_label);


    // Check the correctness of the L3 epilogue loop generated.
    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_epilogue_stmt =
                std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop_epilogue, "l3_epilogue");

    //the body of l3_epilogue loop should consists of 1 statement
    EXPECT_EQ(1, schedule_for_epilogue_stmt->getBody()->getNumStmts());

    auto fir_stmt_epilogue_blk = schedule_for_epilogue_stmt->getBody()->emitFIRNode();
    //expects the statement of the l3 epilogue loop body to be a print statement
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_epilogue_blk->stmts[0]));
    EXPECT_EQ("l1", fir::to<fir::NameNode>(fir_stmt_epilogue_blk->stmts[0])->stmt_label);

}

TEST_F(HighLevelScheduleTest, HighLevelLoopFusionPrologueEpilogue3) {
    istringstream is("func main() "
                             "for i in 1:10; print i; end "
                             "for i in 3:10; print i+1; end "
                             "end");

    fe_->parseStream(is, context_, errors_);
    //attach the labels "l1" and "l2" to the for loop statements
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::ForStmt::Ptr l2_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);
    l1_loop->stmt_label = "l1";
    l2_loop->stmt_label = "l2";

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node = program_schedule_node->fuseForLoop("l1", "l2", "l3");

    main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l3_loop_prologue = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::ForStmt::Ptr l3_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);

    //generate c++ code successfully
    EXPECT_EQ (0,  basicCompileTestWithContext());

    //Check that 2 loops were generated: the prologue and the l3 loop (the main loop)
    EXPECT_EQ (2,  main_func_decl->body->stmts.size());

    // Check the correctness of the L3 loop generated.
    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_stmt =
            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop, "l3");

    //the body of l3 loop should consists of 2 statements
    EXPECT_EQ(2, schedule_for_stmt->getBody()->getNumStmts());

    auto fir_stmt_blk = schedule_for_stmt->getBody()->emitFIRNode();
    //expects both statements of the l3 loop body to be print statements
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[0]));
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_blk->stmts[1]));
    auto fir_name_node = fir::to<fir::NameNode>(fir_stmt_blk->stmts[0]);
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_name_node->body->stmts[0]));
    auto fir_name_node2 = fir::to<fir::NameNode>(fir_stmt_blk->stmts[1]);
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_name_node2->body->stmts[0]));

    // Check the correctness of the L3 prologue loop generated.
    fir::low_level_schedule::ForStmtNode::Ptr schedule_for_prologue_stmt =
                std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_loop_prologue, "l3_prologue");

    //the body of l3_prologue loop should consists of 1 statement
    EXPECT_EQ(1, schedule_for_prologue_stmt->getBody()->getNumStmts());

    auto fir_stmt_prologue_blk = schedule_for_prologue_stmt->getBody()->emitFIRNode();
    //expects the statement of the l3 prologue loop body to be a print statement
    EXPECT_EQ(true, fir::isa<fir::NameNode>(fir_stmt_prologue_blk->stmts[0]));
    EXPECT_EQ("l1", fir::to<fir::NameNode>(fir_stmt_prologue_blk->stmts[0])->stmt_label);

}


TEST_F(HighLevelScheduleTest, SimpleBFSWithPushParallelCASSchedule){
    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "SparsePush")->configApplyParallelization("s1", "dynamic-vertex-parallel")->setApply("s1", "disable_deduplication");
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);

    //check that the apply expr is push and parallel
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(assign_stmt->expr));
    mir::PushEdgeSetApplyExpr::Ptr apply_expr = mir::to<mir::PushEdgeSetApplyExpr>(assign_stmt->expr);
    EXPECT_EQ(true, apply_expr->is_parallel);
}

TEST_F(HighLevelScheduleTest, SimpleBFSWithHyrbidDenseParallelCASSchedule){
    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->setApply("s1", "disable_deduplication");
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);

    //check that the apply expr is push and parallel
    EXPECT_EQ(true, mir::isa<mir::HybridDenseEdgeSetApplyExpr>(assign_stmt->expr));
    mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(assign_stmt->expr);
    EXPECT_EQ(true, apply_expr->is_parallel);
}

TEST_F(HighLevelScheduleTest, SimpleBFSWithHyrbidDenseForwardSerialSchedule){
    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "DensePush-SparsePush")
    ->configApplyParallelization("s1", "serial");
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);

    //check that the apply expr is push and parallel
    EXPECT_EQ(true, mir::isa<mir::HybridDenseForwardEdgeSetApplyExpr>(assign_stmt->expr));
    mir::HybridDenseForwardEdgeSetApplyExpr::Ptr apply_expr = mir::to<mir::HybridDenseForwardEdgeSetApplyExpr>(assign_stmt->expr);
    EXPECT_EQ(false, apply_expr->is_parallel);
}

TEST_F(HighLevelScheduleTest, BFSWithPullParallelSchedule){
    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->setApply("s1", "disable_deduplication");
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);

    //check that the apply expr is push and parallel
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(assign_stmt->expr));
    mir::PullEdgeSetApplyExpr::Ptr apply_expr = mir::to<mir::PullEdgeSetApplyExpr>(assign_stmt->expr);
    EXPECT_EQ(true, apply_expr->is_parallel);
}


TEST_F(HighLevelScheduleTest, SSSPwithHybridDenseForwardSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "DensePush-SparsePush")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}



TEST_F(HighLevelScheduleTest, SSSPwithHybridDenseForwardScheduleNewAPI) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "DensePush-SparsePush")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}


TEST_F(HighLevelScheduleTest, SSSPwithHybridDenseForwardScheduleAsyncMinAPI) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "DensePush-SparsePush")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel");
    istringstream is (sssp_async_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}

TEST_F(HighLevelScheduleTest, SSSPwithHybridDenseSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}


TEST_F(HighLevelScheduleTest, SSSPPullParallelSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}



TEST_F(HighLevelScheduleTest, SSSPPushParallelSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "SparsePush")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}


TEST_F(HighLevelScheduleTest, SSSPPushParallelSlidingQueueSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "SparsePush")->configApplyParallelization("s1", "dynamic-vertex-parallel")->setApply("s1", "sliding_queue");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}


TEST_F(HighLevelScheduleTest, SimpleParallelVertexSetApply){
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() #s1# vertices.apply(addone); print vector_a.sum(); end");

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyParallelization("s1", "dynamic-vertex-parallel");
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}


TEST_F(HighLevelScheduleTest, SimpleSerialVertexSetApply){
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() #s1# vertices.apply(addone); print vector_a.sum(); end");

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyParallelization("s1","serial");
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}

TEST_F(HighLevelScheduleTest, CFPullParallel) {
    istringstream is (cf_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, CFPullParallelLoadBalance) {
    istringstream is (cf_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->setApply("s1", "pull_edge_based_load_balance");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, CFPullParallelLoadBalanceWithGrainSize) {
    istringstream is (cf_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->setApply("s1", "pull_edge_based_load_balance",8000);
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PageRankDeltaPullParallel) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");

    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PageRankDeltaSparsePushParallel) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush")->configApplyParallelization("s1", "dynamic-vertex-parallel");

    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PageRankDeltaPullParallelFuseFields) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PageRankDeltaPullParallelFuseFieldsLoadBalance) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    program->setApply("s1", "pull_edge_based_load_balance")->configApplyDenseVertexSet("s1", "bitvector", "src-vertexset", "DensePull");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PageRankDeltaHybridDenseParallelFuseFieldsLoadBalance) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    program->setApply("s1", "pull_edge_based_load_balance")->configApplyDenseVertexSet("s1", "bitvector", "src-vertexset", "DensePull");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PageRankDeltaDoubleHybridDenseParallelFuseFieldsLoadBalance) {
    istringstream is (prd_double_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel");
    program->setApply("s1", "pull_edge_based_load_balance")->configApplyDenseVertexSet("s1", "bitvector", "src-vertexset", "DensePull");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PRCCPullParallelDifferentSegments) {
    istringstream is (pr_cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("l1:s1", "DensePull")->configApplyParallelization("l1:s1", "dynamic-vertex-parallel")->configApplyNumSSG("l1:s1", "fixed-vertex-count",  10, "DensePull");
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->configApplyNumSSG("s1", "fixed-vertex-count",  20, "DensePull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    // the first apply should be push
    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));
}

TEST_F(HighLevelScheduleTest, PRCCPullParallelTwoEdgesetOneNuma) {
    istringstream is (pr_cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("l1:s1", "DensePull")->configApplyParallelization("l1:s1", "dynamic-vertex-parallel")->configApplyNumSSG("l1:s1", "fixed-vertex-count",  10, "DensePull");
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->configApplyNumSSG("s1", "fixed-vertex-count",  20, "DensePull");
    program->configApplyNUMA("l1:s1", "static-parallel", "DensePull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");

    // the first apply should be push
    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[0]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));
}


TEST_F(HighLevelScheduleTest, BCDefaultSchedule) {
    istringstream is (bc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, BCDensePullSparsePushSchedule) {
    istringstream is (bc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull-SparsePush")
            ->configApplyDenseVertexSet("s1", "bitvector", "src-vertexset", "DensePull")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel")
            ->configApplyDirection("s2", "DensePull-SparsePush")
            ->configApplyDenseVertexSet("s2", "bitvector", "src-vertexset", "DensePull")
            ->configApplyParallelization("s2", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, BCDensePullSparsePushCacheOptimizedSchedule) {
    istringstream is (bc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull-SparsePush")
            ->configApplyDenseVertexSet("s1", "bitvector", "src-vertexset", "DensePull")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel")
            ->configApplyNumSSG("s1", "fixed-vertex-count",  10, "DensePull")
            ->configApplyDirection("s2", "DensePull-SparsePush")
            ->configApplyDenseVertexSet("s2", "bitvector", "src-vertexset", "DensePull")
            ->configApplyParallelization("s2", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, ClosenessCentralityWeightedDefaultSchedule) {
    istringstream is (closeness_centrality_weighted_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}



TEST_F(HighLevelScheduleTest, DeltaSteppingWithEagerPriorityUpdate) {
    istringstream is (delta_stepping_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "eager_priority_update");
    program->configApplyPriorityUpdateDelta("s1", 2);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, DeltaSteppingDensePullParallel) {
    istringstream is (delta_stepping_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull");
    program->configApplyParallelization("s1", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, DeltaSteppingWithDefaultSchedule) {
    istringstream is (delta_stepping_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, DeltaSteppingWithDeltaSparsePushSchedule) {
    istringstream is (delta_stepping_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush");
    program->configApplyParallelization("s1", "dynamic-vertex-parallel");
    program->configApplyPriorityUpdateDelta("s1", 2);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, DeltaSteppingWithEagerPriorityUpdateArgv) {
    istringstream is (delta_stepping_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "eager_priority_update");
    program->configApplyPriorityUpdateDelta("s1", "argv[3]");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, DeltaSteppingWithEagerPriorityUpdateWithMerge) {
    istringstream is (delta_stepping_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "eager_priority_update_with_merge");
    program->configApplyPriorityUpdateDelta("s1", 2);
    program->configBucketMergeThreshold("s1", 1000);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, DeltaSteppingWithEagerPriorityUpdateWithMergeArgv) {
    istringstream is (delta_stepping_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "eager_priority_update_with_merge");
    program->configApplyPriorityUpdateDelta("s1", 2);
    program->configBucketMergeThreshold("s1", "argv[3]");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PPSPDeltaSteppingWithDefaultSchedule) {
    istringstream is (ppsp_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, AStarDeltaSteppingWithDefaultSchedule) {
istringstream is (ppsp_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PPSPDeltaSteppingWithSparsePushParallelSchedule) {
    istringstream is (ppsp_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush");
    program->configApplyParallelization("s1", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PPSPDeltaSteppingWithEagerPriorityUpdateArgv) {
    istringstream is (ppsp_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "eager_priority_update");
    program->configApplyPriorityUpdateDelta("s1", "argv[4]");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PPSPDeltaSteppingWithEagerPriorityUpdateWithMergeArgv) {
    istringstream is (ppsp_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "eager_priority_update_with_merge");
    program->configApplyPriorityUpdateDelta("s1", 2);
    program->configBucketMergeThreshold("s1", "argv[4]");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, AStarDeltaSteppingWithEagerPriorityUpdateWithMergeArgv) {
    istringstream is (astar_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "eager_priority_update_with_merge");
    program->configApplyPriorityUpdateDelta("s1", 2);
    program->configBucketMergeThreshold("s1", "argv[4]");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, ExportPRTest){
    istringstream is (export_pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, ExportPRWithScheduleTest){
    istringstream is (export_pr_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, ExportCFWithScheduleTest){
    istringstream is (export_cf_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "DensePull")
            ->configApplyParallelization("s1", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, UnorderedKCoreDefault){
    istringstream is (unordered_kcore_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, UnorderedKCoreSparsePushParallel){
    istringstream is (unordered_kcore_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush");
    program->configApplyParallelization("s1", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, UnorderedKCoreSparsePushDensePullParallel){
istringstream is (unordered_kcore_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
program->configApplyDirection("s1", "SparsePush-DensePull");
program->configApplyParallelization("s1", "dynamic-vertex-parallel");
EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, KCoreSumReduceBeforeUpdate){
    istringstream is (kcore_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyPriorityUpdate("s1", "constant_sum_reduce_before_update");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, KCoreSparsePushSerial){
    istringstream is (kcore_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, KCoreSparsePushParallel){
istringstream is (kcore_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->configApplyDirection("s1", "SparsePush");
    program->configApplyParallelization("s1", "dynamic-vertex-parallel");
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, KCoreUintSparsePushParallel){
istringstream is (kcore_uint_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
program->configApplyParallelization("s1", "dynamic-vertex-parallel");
EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, KCoreUintSumReduceBeforeUpdate){
istringstream is (kcore_uint_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
program->configApplyPriorityUpdate("s1", "constant_sum_reduce_before_update");
EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, KCoreDensePullSerial){
istringstream is (kcore_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
program->configApplyDirection("s1", "DensePull");
EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, KCoreDensePullParallel){
istringstream is (kcore_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
program->configApplyDirection("s1", "DensePull");
program->configApplyParallelization("s1", "dynamic-vertex-parallel");
EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, KCoreSparsePushDensePullParallel){
istringstream is (kcore_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
program->configApplyDirection("s1", "SparsePush-DensePull");
program->configApplyParallelization("s1", "dynamic-vertex-parallel");
EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, SetCoverUintDefaultSchedule){
istringstream is (setcover_uint_str_);
fe_->parseStream(is, context_, errors_);
fir::high_level_schedule::ProgramScheduleNode::Ptr program
        = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
EXPECT_EQ (0, basicTestWithSchedule(program));
}


