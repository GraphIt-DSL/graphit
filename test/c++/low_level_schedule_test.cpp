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
#include <graphit/frontend/low_level_schedule.h>
#include <graphit/frontend/fir.h>


using namespace std;
using namespace graphit;

class LowLevelScheduleTest : public ::testing::Test {
protected:
    virtual void SetUp(){
        context_ = new graphit::FIRContext();
        errors_ = new std::vector<ParseError>();
        fe_ = new Frontend();
        mir_context_  = new graphit::MIRContext();

    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

        //prints out the MIR, just a hack for now
        //std::cout << "mir: " << std::endl;
        //std::cout << *(mir_context->getStatements().front());
        //std::cout << std::endl;

    }

    /**
     * This test assumes that the fir_context is constructed in the specific test code
     * @return
     */
    bool basicCompileTestWithContext(){
        graphit::Midend* me = new graphit::Midend(context_);
        me->emitMIR(mir_context_);
        graphit::Backend* be = new graphit::Backend(mir_context_);
        return be->emitCPP();
    }

    bool basicTest(std::istream & is){
        fe_->parseStream(is, context_, errors_);
        graphit::Midend* me = new graphit::Midend(context_);

        std::cout << "fir: " << std::endl;
        std::cout << *(context_->getProgram());
        std::cout << std::endl;

        me->emitMIR(mir_context_);
        graphit::Backend* be = new graphit::Backend(mir_context_);
        return be->emitCPP();
    }


    bool basicTestWithSchedule(std::istream & is, Schedule* schedule){
        fe_->parseStream(is, context_, errors_);
        graphit::Midend* me = new graphit::Midend(context_, schedule);
        std::cout << "fir: " << std::endl;
        std::cout << *(context_->getProgram());
        std::cout << std::endl;

        me->emitMIR(mir_context_);
        graphit::Backend* be = new graphit::Backend(mir_context_);
        return be->emitCPP();
    }

    std::vector<ParseError> * errors_;
    graphit::FIRContext* context_;
    Frontend * fe_;
    graphit::MIRContext* mir_context_;
};


TEST_F(LowLevelScheduleTest, SimpleStructSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vector_b : vector{Vertex}(float) = 0.0;\n"
    );
    Schedule *schedule = new Schedule();
    FieldVectorPhysicalDataLayout vector_a_layout = {"vector_a", FieldVectorDataLayoutType::STRUCT, "struct_a_b"};
    FieldVectorPhysicalDataLayout vector_b_layout = {"vector_b", FieldVectorDataLayoutType::STRUCT, "struct_a_b"};
    auto physical_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0, basicTestWithSchedule(is, schedule));

}

TEST_F(LowLevelScheduleTest, AddoneWithNoSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");

    EXPECT_EQ (0, basicTest(is));
}

TEST_F(LowLevelScheduleTest, AddoneWithArraySchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");
    Schedule *schedule = new Schedule();
    FieldVectorPhysicalDataLayout vector_a_layout = {"vector_a", FieldVectorDataLayoutType::ARRAY, ""};
    FieldVectorPhysicalDataLayout vector_b_layout = {"vector_b", FieldVectorDataLayoutType::ARRAY, ""};
    auto physical_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0, basicTestWithSchedule(is, schedule));
}

TEST_F(LowLevelScheduleTest, AddoneWithStructSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");

    // constructs a schedule object that fuses vector_a and vector_b into an array of struct
    Schedule *schedule = new Schedule();
    FieldVectorPhysicalDataLayout vector_a_layout = {"vector_a", FieldVectorDataLayoutType::STRUCT, "struct_a_b"};
    FieldVectorPhysicalDataLayout vector_b_layout = {"vector_b", FieldVectorDataLayoutType::STRUCT, "struct_a_b"};
    auto physical_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0, basicTestWithSchedule(is, schedule));
}


TEST_F(LowLevelScheduleTest, SimpleEdgesetApplyPullSchedule) {
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


TEST_F(LowLevelScheduleTest, SimpleEdgesetApplyPushSchedule) {
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


TEST_F(LowLevelScheduleTest, SimpleEdgesetApplyNestedLabelsPullSchedule) {
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

TEST_F(LowLevelScheduleTest, SimpleForEdgesetApplyNoNestedLabelsPullSchedule) {
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


TEST_F(LowLevelScheduleTest, SimpleBFSPushSchedule) {
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
 * Tests the basic loop cloning API
 */
TEST_F(LowLevelScheduleTest, SimpleLoopBodyCloning) {
    istringstream is("func main() for i in 1:10; print i; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    EXPECT_EQ (1,  l1_body_blk->getNumStmts());
}

/**
 * Tests the basic loop cloning API
 */
TEST_F(LowLevelScheduleTest, RemoveLabelFail) {
    istringstream is("func main() for i in 1:10; print i; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);


    EXPECT_EQ (false,  schedule_program_node->removeLabelNode("l2"));
}



/**
 * Tests cloning a more complicated loop body with expr stmt and apply expr
 */
TEST_F(LowLevelScheduleTest, LoopBodyApplyCloning) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                             "for i in 1:10 edges.apply(updateEdge); end\n"
                             "end\n"
    );
    Schedule * schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType ::PULL};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();

    //We are constructing a nested scope label this time
    (*apply_schedules)["l1:s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //set the label of the edgeset apply expr stamt
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ForStmt::Ptr for_stmt = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    for_stmt->stmt_label = "l1";

    //set the label of the for loop
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(for_stmt->body->stmts[0]);
    apply_stmt->stmt_label = "s1";
    graphit::Midend* me = new graphit::Midend(context_, schedule);
    me->emitMIR(mir_context_);
    graphit::Backend* be = new graphit::Backend(mir_context_);

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    EXPECT_EQ (1,  l1_body_blk->getNumStmts());

    EXPECT_EQ (true, fir::isa<fir::ExprStmt>(
            l1_body_blk->emitFIRNode()->stmts[0]));

}

/**
 * Tests cloning a more complicated loop body with expr stmt and apply expr
 */
TEST_F(LowLevelScheduleTest, LoopBodyApplyCloningFail) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                             "for i in 1:10 edges.apply(updateEdge); end\n"
                             "end\n"
    );

    fe_->parseStream(is, context_, errors_);

    // use the low level scheduling API to make clone the body of "l2" loop
    // there is no l2 loop, so we expct a null pointer
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l2");

    EXPECT_EQ (nullptr,  l1_body_blk);
}

/**
 * Test for building a for loop scheduling node
 */
TEST_F(LowLevelScheduleTest, SimpleLoopNodeCreation) {
    istringstream is("func main() for i in 1:10; print i; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    //create and set bounds for l2_loop
    fir::low_level_schedule::RangeDomain::Ptr l2_range_domain
            = std::make_shared<fir::low_level_schedule::RangeDomain>(0, 2);

    //create a new loop node with labels "l2"
    fir::low_level_schedule::ForStmtNode::Ptr l2_loop
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(l2_range_domain, "l2");

    //append the original l1_loop body to l2_loop
    l2_loop->appendLoopBody(l1_body_blk);

    EXPECT_EQ (1,  l2_loop->getBody()->getNumStmts());
    EXPECT_EQ (true,  fir::isa<fir::PrintStmt>(
            l2_loop->getBody()->emitFIRNode()->stmts[0]));

}

/**
 * Test for building a scheduling name node
 */
TEST_F(LowLevelScheduleTest, SimpleNameNodeCreation) {
    istringstream is("func main() for i in 1:10; print i; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    //create a new name node with labels "l1"
    fir::low_level_schedule::NameNode::Ptr l1_name
            = std::make_shared<fir::low_level_schedule::NameNode>(l1_body_blk, "l1");


    EXPECT_EQ (1,  l1_name->getBody()->getNumStmts());
    EXPECT_EQ (true,  fir::isa<fir::PrintStmt>(
            l1_name->getBody()->emitFIRNode()->stmts[0]));

}


TEST_F(LowLevelScheduleTest, SimpleInsertNameNodeBefore) {
    istringstream is("func main() for i in 1:2; print 4; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    //create a new name node with labels "l1"
    fir::low_level_schedule::NameNode::Ptr l1_name
            = std::make_shared<fir::low_level_schedule::NameNode>(l1_body_blk, "l1");

    schedule_program_node->insertBefore(l1_name, "l1");

    //check if the namenode has been inserted by checking the number of stmts in main func
    EXPECT_EQ (2,  main_func_decl->body->stmts.size());

    //check if the namenode has been inserted by checking the type of the first stmt
    EXPECT_EQ (true,  fir::isa<fir::NameNode>(
            main_func_decl->body->stmts[0]));

    EXPECT_EQ (0,  basicCompileTestWithContext());

}

TEST_F(LowLevelScheduleTest, SimpleInsertNameNodeBeforeFail) {
    istringstream is("func main() for i in 1:2; print 4; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    //create a new name node with labels "l1"
    fir::low_level_schedule::NameNode::Ptr l1_name
            = std::make_shared<fir::low_level_schedule::NameNode>(l1_body_blk, "l1");



    //check if the namenode has been inserted by checking the number of stmts in main func
    EXPECT_EQ (false,  schedule_program_node->insertBefore(l1_name, "l2"));


}

TEST_F(LowLevelScheduleTest, SimpleInsertForLoopNodeBefore) {
    istringstream is("func main() for i in 1:2; print 4; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    //create and set bounds for l2_loop
    fir::low_level_schedule::RangeDomain::Ptr l2_range_domain
            = std::make_shared<fir::low_level_schedule::RangeDomain>(0, 2);

    //create a new name node with labels "l1"
    fir::low_level_schedule::ForStmtNode::Ptr l2_loop
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(
                    l2_range_domain, l1_body_blk, "l2", "i");

    schedule_program_node->insertBefore(l2_loop, "l1");

    //check if the for loop has been inserted by checking the number of stmts in main func
    EXPECT_EQ (2,  main_func_decl->body->stmts.size());

    //check if the for loop has been inserted by checking the type of the first stmt
    EXPECT_EQ (true,  fir::isa<fir::ForStmt>(
            main_func_decl->body->stmts[0]));

    EXPECT_EQ (0,  basicCompileTestWithContext());

}


TEST_F(LowLevelScheduleTest, AppendLoopBody) {
    istringstream is("func main() for i in 1:2; print 4; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");

    //create and set bounds for l2_loop
    fir::low_level_schedule::RangeDomain::Ptr l2_range_domain
            = std::make_shared<fir::low_level_schedule::RangeDomain>(0, 2);

    //create a new name node with labels "l1"
    fir::low_level_schedule::ForStmtNode::Ptr l2_loop
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(
                    l2_range_domain, l1_body_blk, "l2", "i");

    l2_loop->appendLoopBody(l1_body_blk);

    schedule_program_node->insertBefore(l2_loop, "l1");

    //check if the for loop has been inserted by checking the number of stmts in main func
    EXPECT_EQ (2,  main_func_decl->body->stmts.size());

    //test to see if the statement has been inserted into the body of the loop
    EXPECT_EQ (2,  l2_loop->getBody()->getNumStmts());

    //check if the for loop has been inserted by checking the type of the first stmt
    EXPECT_EQ (true,  fir::isa<fir::ForStmt>(
            main_func_decl->body->stmts[0]));

}


/**
 * A test case that tries to break the 10 iters loop into a 2 iters and a 8 iters loop
 */
TEST_F(LowLevelScheduleTest, SimpleLoopIndexSplit) {
    istringstream is("func main() for i in 1:10; print i; end end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    l1_loop->stmt_label = "l1";

    // use the low level scheduling API to make clone the body of "l1" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");


    //create and set bounds for l2_loop and l3_loop
    fir::low_level_schedule::RangeDomain::Ptr l2_range_domain
        = std::make_shared<fir::low_level_schedule::RangeDomain>(0, 2);
    fir::low_level_schedule::RangeDomain::Ptr l3_range_domain
            = std::make_shared<fir::low_level_schedule::RangeDomain>(2, 10);

    //create two new fissed loop node with labels "l2" and "l3"
    fir::low_level_schedule::ForStmtNode::Ptr l2_loop
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(l2_range_domain, l1_body_blk,  "l2", "i");
    fir::low_level_schedule::ForStmtNode::Ptr l3_loop
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_range_domain, l1_body_blk, "l3", "i");


    //insert l2_loop and l3_loop back into the program right before l1_loop
    schedule_program_node->insertBefore(l2_loop, "l1");
    schedule_program_node->insertBefore(l3_loop, "l1");


    //remove l1_loop
    schedule_program_node->removeLabelNode("l1");

    //generate c++ code successfully
    EXPECT_EQ (0,  basicCompileTestWithContext());

    //expects two loops in the main function decl
    EXPECT_EQ (2,  main_func_decl->body->stmts.size());

}


TEST_F(LowLevelScheduleTest, SimpleLoopFusion) {
    istringstream is("func main() "
                             "for i in 1:10; print i; end "
                             "for i in 1:10; print i+1; end "
                             "end");

    fe_->parseStream(is, context_, errors_);
    //attach a label "l1" to the for stataement
    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::ForStmt::Ptr l2_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[1]);

    l1_loop->stmt_label = "l1";
    l2_loop->stmt_label = "l2";

    // use the low level scheduling API to make clone the body of "l1" and "l2" loop
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
            = schedule_program_node->cloneLabelLoopBody("l1");
    fir::low_level_schedule::StmtBlockNode::Ptr l2_body_blk
            = schedule_program_node->cloneLabelLoopBody("l2");

    //create and set bounds for l2_loop and l3_loop
    fir::low_level_schedule::RangeDomain::Ptr l3_range_domain
            = std::make_shared<fir::low_level_schedule::RangeDomain>(1, 10);

    fir::low_level_schedule::ForStmtNode::Ptr l3_loop
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_range_domain, l1_body_blk,  "l3", "i");

    l3_loop->appendLoopBody(l2_body_blk);
    schedule_program_node->insertBefore(l3_loop, "l1");
    schedule_program_node->removeLabelNode("l1");
    schedule_program_node->removeLabelNode("l2");

    //generate c++ code successfully
    EXPECT_EQ (0,  basicCompileTestWithContext());

    //ony l3 loop statement left
    EXPECT_EQ (1,  main_func_decl->body->stmts.size());

    //the body of l3 loop should consists of 2 statements
    EXPECT_EQ(2, l3_loop->getBody()->getNumStmts());

    auto fir_stmt_blk = l3_loop->getBody()->emitFIRNode();
    //expects both statements of the l3 loop body to be print statements
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_stmt_blk->stmts[0]));
    EXPECT_EQ(true, fir::isa<fir::PrintStmt>(fir_stmt_blk->stmts[1]));

}

TEST_F(LowLevelScheduleTest, SimpleFunctionFusion) {
    istringstream is(        "func paddone (a : int) print a + 1; end\n"
                             "func paddtwo (a : int) print a + 2; end\n"
                             "func main() \n"
                                "paddone(4, 5); \n"
                             "end");
    fe_->parseStream(is, context_, errors_);

    fir::FuncDecl::Ptr pddone = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::FuncDecl::Ptr pddtwo = fir::to<fir::FuncDecl>(context_->getProgram()->elems[1]);
    fir::FuncDecl::Ptr main_func = fir::to<fir::FuncDecl>(context_->getProgram()->elems[2]);

    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::FuncDeclNode::Ptr fused_func
            = schedule_program_node->cloneFuncDecl("paddone");
    EXPECT_NE(nullptr, fused_func);

    fused_func->setFunctionName("fused_func");
    fused_func->appendFuncDeclBody(schedule_program_node->cloneFuncBody("paddtwo"));
    schedule_program_node->insertAfter(fused_func, "paddtwo");

    // Expects that the program still compiles
    EXPECT_EQ (0,  basicCompileTestWithContext());

    // Expects four function declarations now
    EXPECT_EQ (4,  context_->getProgram()->elems.size());

    EXPECT_NE(nullptr, fused_func->getBody());

    // Expects two stmts in the fused function
    EXPECT_EQ(2, fused_func->getBody()->getNumStmts());


}


TEST_F(LowLevelScheduleTest, SimpleApplyFunctionFusion) {

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

    //use the low level APIs to fuse the apply functions
    fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
            = std::make_shared<fir::low_level_schedule::ProgramNode>(context_);
    fir::low_level_schedule::ApplyNode::Ptr first_apply_node
            = schedule_program_node->cloneApplyNode("l1");
    fir::low_level_schedule::ApplyNode::Ptr second_apply_node
            = schedule_program_node->cloneApplyNode("l2");
    first_apply_node->updateStmtLabel("l3");

    // create a fused function
    fir::low_level_schedule::FuncDeclNode::Ptr first_apply_func_decl
            = schedule_program_node->cloneFuncDecl(first_apply_node->getApplyFuncName());
    fir::low_level_schedule::StmtBlockNode::Ptr second_apply_func_body
            = schedule_program_node->cloneFuncBody(second_apply_node->getApplyFuncName());

    first_apply_func_decl->appendFuncDeclBody(second_apply_func_body);
    first_apply_func_decl->setFunctionName("fused_func");

    schedule_program_node->insertAfter(first_apply_func_decl, second_apply_node->getApplyFuncName());

    first_apply_node->updateApplyFunc("fused_func");

    schedule_program_node->insertBefore(first_apply_node, "l2");
    schedule_program_node->removeLabelNode("l1");
    schedule_program_node->removeLabelNode("l2");

    // Expects that the program still compiles
    EXPECT_EQ (0,  basicCompileTestWithContext());

    // Expects one more fused function declarations now
    EXPECT_EQ (9,  context_->getProgram()->elems.size());

    // Expects two stmts in the fused function
    EXPECT_EQ(2, first_apply_func_decl->getBody()->getNumStmts());

}