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
#include <graphit/frontend/low_level_schedule.h.h>
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
            l1_body_blk->getFirStmtBlk()->stmts[0]));

}

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
            l2_loop->getBody()->getFirStmtBlk()->stmts[0]));

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
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(l2_range_domain, "l2");
    fir::low_level_schedule::ForStmtNode::Ptr l3_loop
            = std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_range_domain, "l3");


    //append the original l1_loop body to l2_loop and l3_loop
    l2_loop->appendLoopBody(l1_body_blk);
    l3_loop->appendLoopBody(l1_body_blk);

    //insert l2_loop and l3_loop back into the program right before l1_loop
    schedule_program_node->insertBefore(l2_loop, "l1");
    schedule_program_node->insertBefore(l3_loop, "l1");


    //remove l1_loop
    schedule_program_node->removeLabelNode("l1");

    EXPECT_EQ (0,  basicTest(is));
}
