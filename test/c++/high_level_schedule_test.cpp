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
        bfs_is_ = istringstream("element Vertex end\n"
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
                                        "      #s1# frontier = edges.from(frontier).to(toFilter).apply(updateEdge).modified(parent); "
                                        "  end\n"
                                        "  print \"finished running BFS\"; \n"
                                        "end");

        pr_is_ = istringstream("element Vertex end\n"
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
                                       "        vertices.apply(updateVertex);\n"
                                       "        print error.sum();"
                                       "    end\n"
                                       "end"
        );
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
    istringstream bfs_is_;
    istringstream pr_is_;

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

//TODO: add test cases for loop fusion and apply function fusion



TEST_F(HighLevelScheduleTest, BFSPushSchedule) {
    fe_->parseStream(bfs_is_, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->setApply("s1", "push");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::WhileStmt::Ptr while_stmt = mir::to<mir::WhileStmt>((*(main_func_decl->body->stmts))[2]);
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>((*(while_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(assign_stmt->expr));
}

TEST_F(HighLevelScheduleTest, BFSPullSchedule) {
    fe_->parseStream(bfs_is_, context_, errors_);
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

TEST_F(HighLevelScheduleTest, PRNestedSchedule) {
    fe_->parseStream(pr_is_, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);
    // The schedule does a array of SoA optimization, and split the loops
    // while supplying different schedules for the two splitted loops
    program->fuseFields("old_rank", "out_degrees")->splitForLoop("l1", "l2", "l3", 2, 8);
    program->setApply("l2:s1", "push")->setApply("l3:s1", "pull");
    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));

    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    // 1 for the conversion from AoS to SoA, 2 for the splitted for loops
    EXPECT_EQ(3, main_func_decl->body->stmts->size());

    // the first apply should be push
    mir::ForStmt::Ptr for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[1]);
    mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PushEdgeSetApplyExpr>(expr_stmt->expr));

    // the second apply should be pull
    for_stmt = mir::to<mir::ForStmt>((*(main_func_decl->body->stmts))[2]);
    expr_stmt = mir::to<mir::ExprStmt>((*(for_stmt->body->stmts))[0]);
    EXPECT_EQ(true, mir::isa<mir::PullEdgeSetApplyExpr>(expr_stmt->expr));

}



TEST_F(HighLevelScheduleTest, BFSSerialPushSparseSchedule) {
    fe_->parseStream(bfs_is_, context_, errors_);
    fir::high_level_schedule::ProgramScheduleNode::Ptr program
            = std::make_shared<fir::high_level_schedule::ProgramScheduleNode>(context_);

    program->setApply("s1", "push");
    program->setVertexSet("frontier", "sparse");
    program->setApply("s1", "sparse_frontier");

    //generate c++ code successfully
    EXPECT_EQ (0, basicTestWithSchedule(program));
    mir::FuncDecl::Ptr main_func_decl = mir_context_->getFunction("main");
    mir::VarDecl::Ptr frontier_decl = mir::to<mir::VarDecl>((*(main_func_decl->body->stmts))[0]);
    mir::VertexSetAllocExpr::Ptr alloc_expr = mir::to<mir::VertexSetAllocExpr>(frontier_decl->initVal);
    EXPECT_EQ(mir::VertexSetAllocExpr::Layout::SPARSE, alloc_expr->layout);
}