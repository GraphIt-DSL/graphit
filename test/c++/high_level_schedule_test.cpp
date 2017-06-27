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

using namespace std;
using namespace graphit;

class HighLevelScheduleTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        context_ = new graphit::FIRContext();
        errors_ = new std::vector<ParseError>();
        fe_ = new Frontend();
        mir_context_ = new graphit::MIRContext();

    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

        //prints out the MIR, just a hack for now
        //std::cout << "mir: " << std::endl;
        //std::cout << *(mir_context->getStatements().front());
        //std::cout << std::endl;

    }

    bool basicTest(std::istream &is) {
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
    bool basicCompileTestWithContext() {
        graphit::Midend *me = new graphit::Midend(context_);
        me->emitMIR(mir_context_);
        graphit::Backend *be = new graphit::Backend(mir_context_);
        return be->emitCPP();
    }


    bool basicTestWithSchedule(std::istream &is, Schedule *schedule) {
        fe_->parseStream(is, context_, errors_);
        graphit::Midend *me = new graphit::Midend(context_, schedule);
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
};

TEST_F(HighLevelScheduleTest, SimpleStructSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vector_b : vector{Vertex}(float) = 0.0;\n"
    );
    Schedule *schedule = new Schedule();
    PhysicalDataLayout vector_a_layout = {"vector_a", DataLayoutType::STRUCT, "struct_a_b"};
    PhysicalDataLayout vector_b_layout = {"vector_b", DataLayoutType::STRUCT, "struct_a_b"};
    auto physical_layouts = new std::map<std::string, PhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0, basicTestWithSchedule(is, schedule));

}

TEST_F(HighLevelScheduleTest, AddoneWithNoSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");

    EXPECT_EQ (0, basicTest(is));
}

TEST_F(HighLevelScheduleTest, AddoneWithArraySchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");
    Schedule *schedule = new Schedule();
    PhysicalDataLayout vector_a_layout = {"vector_a", DataLayoutType::ARRAY, ""};
    PhysicalDataLayout vector_b_layout = {"vector_b", DataLayoutType::ARRAY, ""};
    auto physical_layouts = new std::map<std::string, PhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0, basicTestWithSchedule(is, schedule));
}

TEST_F(HighLevelScheduleTest, AddoneWithStructSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");

    // constructs a schedule object that fuses vector_a and vector_b into an array of struct
    Schedule *schedule = new Schedule();
    PhysicalDataLayout vector_a_layout = {"vector_a", DataLayoutType::STRUCT, "struct_a_b"};
    PhysicalDataLayout vector_b_layout = {"vector_b", DataLayoutType::STRUCT, "struct_a_b"};
    auto physical_layouts = new std::map<std::string, PhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0, basicTestWithSchedule(is, schedule));
}


TEST_F(HighLevelScheduleTest, SimpleEdgesetApplyPullSchedule) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                             "edges.apply(updateEdge); \n"
                             "end\n"
    );
    Schedule *schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType::PULL};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //auto edges_apply_stmt = context_->getProgram()->elems[]
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(main_func_decl->body->stmts[0]);
    apply_stmt->stmt_label = "s1";

    graphit::Midend *me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend *be = new graphit::Backend(mir_context_);


    EXPECT_EQ (0, be->emitCPP());

}


TEST_F(HighLevelScheduleTest, SimpleEdgesetApplyPushSchedule) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                             "edges.apply(updateEdge); \n"
                             "end\n"
    );
    Schedule *schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType::PUSH};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //auto edges_apply_stmt = context_->getProgram()->elems[]
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(main_func_decl->body->stmts[0]);
    apply_stmt->stmt_label = "s1";

    graphit::Midend *me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend *be = new graphit::Backend(mir_context_);


    EXPECT_EQ (0, be->emitCPP());

}


TEST_F(HighLevelScheduleTest, SimpleEdgesetApplyNestedLabelsPullSchedule) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                             "for i in 1:10 edges.apply(updateEdge); end\n"
                             "end\n"
    );
    Schedule *schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType::PULL};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();

    //We are constructing a nested scope label this time
    (*apply_schedules)["l1:s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //set the label of the edgeset apply expr stamt
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ForStmt::Ptr for_stmt = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    for_stmt->stmt_label = "l1";

    //set the label of the for loop
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(for_stmt->body->stmts[0]);
    apply_stmt->stmt_label = "s1";

    graphit::Midend *me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend *be = new graphit::Backend(mir_context_);


    EXPECT_EQ (0, be->emitCPP());

}

TEST_F(HighLevelScheduleTest, SimpleForEdgesetApplyNoNestedLabelsPullSchedule) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                             "for i in 1:10 edges.apply(updateEdge); end\n"
                             "end\n"
    );
    Schedule *schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType::PULL};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();

    //We are constructing a nested scope label this time
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //set the label of the edgeset apply expr stamt
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ForStmt::Ptr for_stmt = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);

    //set the label of the for loop
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(for_stmt->body->stmts[0]);
    apply_stmt->stmt_label = "s1";

    graphit::Midend *me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend *be = new graphit::Backend(mir_context_);


    EXPECT_EQ (0, be->emitCPP());

}


TEST_F(HighLevelScheduleTest, SimpleBFSPushSchedule) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const parent : vector{Vertex}(int) = -1;\n"
                             "func updateEdge(src : Vertex, dst : Vertex) -> output : bool "
                             "parent[dst] = src; "
                             "output = true; "
                             "end\n"
                             "func toFilter(v : Vertex) -> output : bool "
                             "output = parent[v] == -1; "
                             "end\n"
                             "func main() "
                             "var frontier : vertexset{Vertex} = new vertexset{Vertex}(0); "
                             "frontier.addVertex(1); "
                             "while (frontier.getVertexSetSize() != 0) "
                             "frontier = edges.from(frontier).to(toFilter).apply(updateEdge); "
                             "end\n"
                             "print \"finished running BFS\"; \n"
                             "end");
    Schedule *schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType::PUSH};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();

    //We are constructing a nested scope label this time
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //set the label of the edgeset apply expr stamt
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[7]);
    fir::WhileStmt::Ptr while_stmt = fir::to<fir::WhileStmt>(main_func_decl->body->stmts[2]);

    //set the label of the for loop
    fir::AssignStmt::Ptr assign_stmt = fir::to<fir::AssignStmt>(while_stmt->body->stmts[0]);
    assign_stmt->stmt_label = "s1";

    graphit::Midend *me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend *be = new graphit::Backend(mir_context_);

    ofstream test_file;
    test_file.open("../test.cpp");
    be->emitCPP(test_file);
    test_file.close();
    std::cout << exec_cmd("g++ -std=c++11 -I ../../src/runtime_lib/ ../test.cpp  -o test.o 2>&1");
    std::cout << exec_cmd("./test.o 2>&1");

    EXPECT_EQ (0, 0);
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
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_stmt_blk->stmts[0]));
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_stmt_blk->stmts[1]));
}

//TODO: add test cases for loop fusion and apply function fusion
