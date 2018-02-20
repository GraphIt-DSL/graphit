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
                                "      #s1# frontier = edges.from(frontier).to(toFilter).applyModified(updateEdge, parent); "
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



        bfs_str_ =  string (bfs_char);
        pr_str_ = string(pr_char);
        sssp_str_ = string  (sssp_char);
        cf_str_ = string  (cf_char);
        cc_str_ = string  (cc_char);
        prd_str_ = string  (prd_char);
        prd_double_str_ = string  (prd_double_char);

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
    string cf_str_;
    string cc_str_;
    string prd_str_;
    string prd_double_str_;



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
    program_schedule_node->setApply("l3:l1:s1", "parallel");
    // Expects that the program still compiles
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));

}

TEST_F(HighLevelScheduleTest, BFSPushSerialSchedule) {
    istringstream is (bfs_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->setApply("s1", "push")->setApply("s1", "disable_deduplication");
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

    program->setApply("s1", "push")->setApply("s1", "parallel")->setApply("s1", "disable_deduplication");
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

    program->setApply("s1", "push")->setApply("s1", "sliding_queue")->setApply("s1", "parallel")->setApply("s1", "disable_deduplication");
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

    program->setApply("s1", "pull");
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

    program->setApply("s1", "hybrid_dense")->setApply("s1", "parallel")->setApply("s1", "disable_deduplication");
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

TEST_F(HighLevelScheduleTest, CCHybridDenseSchedule) {
    istringstream is (cc_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->setApply("s1", "hybrid_dense")->setApply("s1", "parallel");
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

    program->setApply("s1", "hybrid_dense", 2)->setApply("s1", "parallel");
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

    program->setApply("s1", "hybrid_dense")->setApply("s1", "parallel")->setApply("s1", "pull_frontier_bitvector");
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

    program->configApplyDirection("s1", "hybrid_dense")->configApplyParallelization("s1", "parallel")
            ->configApplyDataStructure("s1", "pull_frontier_bitvector");
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

    program->setApply("s1", "pull")->setApply("s1", "parallel");
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

    program->setApply("s1", "push")->setApply("s1", "parallel");
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
    program->setApply("l1:s1", "pull")->setApply("l1:s1", "parallel");
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
    program->setApply("l1:s1", "pull", 2)->setApply("l1:s1", "parallel");
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
    program->setApply("l1:s1", "pull")->setApply("l1:s1", "parallel");
    program->setApply("l1:s2", "parallel");

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
    program->setApply("l1:s1", "push")->setApply("l1:s1", "parallel");
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
    program_schedule_node->setApply("s1", "push")->setApply("s1", "parallel")->setApply("s1", "disable_deduplication");
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
    program_schedule_node->setApply("s1", "hybrid_dense")->setApply("s1", "parallel")->setApply("s1", "disable_deduplication");
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

TEST_F(HighLevelScheduleTest, BFSWithPullParallelSchedule){
    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->setApply("s1", "pull")->setApply("s1", "parallel")->setApply("s1", "disable_deduplication");
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
    program_schedule_node->setApply("s1", "hybrid_dense_forward")->setApply("s1", "parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);

    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}

TEST_F(HighLevelScheduleTest, SSSPwithHybridDenseForwardScheduleNewAPI) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->configApplyDirection("s1", "hybrid_dense_forward")->configApplyParallelization("s1", "parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}

TEST_F(HighLevelScheduleTest, SSSPwithHybridDenseSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->setApply("s1", "hybrid_dense")->setApply("s1", "parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}


TEST_F(HighLevelScheduleTest, SSSPPullParallelSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->setApply("s1", "pull")->setApply("s1", "parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}



TEST_F(HighLevelScheduleTest, SSSPPushParallelSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->setApply("s1", "push")->setApply("s1", "parallel");
    istringstream is (sssp_str_);
    fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0,  basicTestWithSchedule(program_schedule_node));
}


TEST_F(HighLevelScheduleTest, SSSPPushParallelSlidingQueueSchedule) {

    fir::high_level_schedule::ProgramScheduleNode::Ptr program_schedule_node
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program_schedule_node->setApply("s1", "push")->setApply("s1", "parallel")->setApply("s1", "sliding_queue");
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
    program_schedule_node->setApply("s1", "parallel");
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
    program_schedule_node->setApply("s1", "serial");
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
    program->setApply("s1", "pull")->setApply("s1", "parallel");
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
    program->setApply("s1", "pull")->setApply("s1", "parallel")->setApply("s1", "pull_edge_based_load_balance");
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
    program->setApply("s1", "pull")->setApply("s1", "parallel")->setApply("s1", "pull_edge_based_load_balance",8000);
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PageRankDeltaPullParallel) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->setApply("s1", "pull")->setApply("s1", "parallel");

    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PageRankDeltaPullParallelFuseFields) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->setApply("s1", "pull")->setApply("s1", "parallel");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PageRankDeltaPullParallelFuseFieldsLoadBalance) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->setApply("s1", "pull")->setApply("s1", "parallel");
    program->setApply("s1", "pull_edge_based_load_balance")->setApply("s1", "pull_frontier_bitvector");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

TEST_F(HighLevelScheduleTest, PageRankDeltaHybridDenseParallelFuseFieldsLoadBalance) {
    istringstream is (prd_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->setApply("s1", "hybrid_dense")->setApply("s1", "parallel");
    program->setApply("s1", "pull_edge_based_load_balance")->setApply("s1", "pull_frontier_bitvector");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}


TEST_F(HighLevelScheduleTest, PageRankDeltaDoubleHybridDenseParallelFuseFieldsLoadBalance) {
    istringstream is (prd_double_str_);
    fe_->parseStream(is, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    program->setApply("s1", "hybrid_dense")->setApply("s1", "parallel");
    program->setApply("s1", "pull_edge_based_load_balance")->setApply("s1", "pull_frontier_bitvector");
    program->fuseFields("delta", "out_degree");
    // generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
}

