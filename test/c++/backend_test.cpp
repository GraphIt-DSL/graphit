//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/midend/midend.h>
#include <graphit/backend/backend.h>
#include <graphit/frontend/error.h>
#include <graphit/utils/exec_cmd.h>

using namespace std;
using namespace graphit;

class BackendTest : public ::testing::Test {
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

//tests back end
TEST_F(BackendTest, SimpleVarDecl) {
    istringstream is("const a : int = 3 + 4;");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionDecl) {
    istringstream is("func add(a : int, b: int) -> c : int  end");
    EXPECT_EQ (0 ,  basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionDeclWithNoReturn) {
    istringstream is("func add(a : int, b: int)  end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionWithVarDecl) {
    istringstream is("func add(a : int, b: int) -> c : int var d : int = 3; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionWithAdd) {
    istringstream is("func add(a : int, b: int) -> c : int c = a + b; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(BackendTest, MainFunctionWithPrint) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(BackendTest, MainFunctionWithCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, MainFunctionWithPrintCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() print add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, ElementDecl) {
    istringstream is("element Vertex end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleVertexSetDeclAlloc) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleVertexSetDeclAllocWithMain) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() print 4; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleMultiArrayAllocWithMain) {
    istringstream is("element Vertex end\n"
                             "const old_rank : vector{Vertex}(float) = 0.0;\n"
                             "const new_rank : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleEdgeSetWithMain) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "func main() print 0; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleVariable){
    istringstream is("func main() var a : int = 4; print a; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleVectorSum){
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() var sum : float = vector_a.sum(); print sum; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleVertexSetApply){
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleEdgeSetLoad){
    istringstream is("element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "func main() print 0; end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(BackendTest, SimpleVertexSetLoad){
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "func main() print 0; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleEdgeSetApply) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "func srcAddOne(src : Vertex, dst : Vertex) "
                             "vector_a[src] = vector_a[src] + 1; end\n"
                             "func main() edges.apply(srcAddOne); print vector_a.sum(); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleForLoops) {
    istringstream is("func main() for i in 1:10; print i; end end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, VertexSetGetSize) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const size : int = vertices.size();\n"
                             "func main() print size; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, EdgeSetGetOutDegrees) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                             "func main() print out_degrees.sum(); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleFixedIterPageRank) {
    istringstream is("element Vertex end\n"
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
                             "    new_rank[dst] = old_rank[src] / out_degrees[src];\n"
                             "end\n"
                             "func updateVertex(v : Vertex)\n"
                             "    new_rank[v] = beta_score + damp*(new_rank[v]);\n"
                             "    error[v]    = fabs ( new_rank[v] - old_rank[v]);\n"
                             "    old_rank[v] = new_rank[v];\n"
                             "    new_rank[v] = 0.0;\n"
                             "end\n"
                             "func main()\n"
                             "    for i in 1:10\n"
                             "        edges.apply(updateEdge);\n"
                             "        vertices.apply(updateVertex);\n"
                             "        print error.sum();"
                             "    end\n"
                             "end"
    );
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(BackendTest, SimpleStructSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vector_b : vector{Vertex}(float) = 0.0;\n"
    );
    Schedule * schedule = new Schedule();
    PhysicalDataLayout vector_a_layout = {"vector_a", DataLayoutType::STRUCT, "struct_a_b"};
    PhysicalDataLayout vector_b_layout = {"vector_b", DataLayoutType::STRUCT, "struct_a_b"};
    auto physical_layouts = new std::map<std::string, PhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0,  basicTestWithSchedule(is, schedule));

}

TEST_F(BackendTest, AddoneWithNoSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");

    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, AddoneWithArraySchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");
    Schedule * schedule = new Schedule();
    PhysicalDataLayout vector_a_layout = {"vector_a", DataLayoutType::ARRAY, ""};
    PhysicalDataLayout vector_b_layout = {"vector_b", DataLayoutType::ARRAY, ""};
    auto physical_layouts = new std::map<std::string, PhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0,  basicTestWithSchedule(is, schedule));
}

TEST_F(BackendTest, AddoneWithStructSchedule) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vector_b : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");

    // constructs a schedule object that fuses vector_a and vector_b into an array of struct
    Schedule * schedule = new Schedule();
    PhysicalDataLayout vector_a_layout = {"vector_a", DataLayoutType::STRUCT, "struct_a_b"};
    PhysicalDataLayout vector_b_layout = {"vector_b", DataLayoutType::STRUCT, "struct_a_b"};
    auto physical_layouts = new std::map<std::string, PhysicalDataLayout>();
    (*physical_layouts)["vector_a"] = vector_a_layout;
    (*physical_layouts)["vector_b"] = vector_b_layout;

    schedule->physical_data_layouts = physical_layouts;
    EXPECT_EQ (0,  basicTestWithSchedule(is, schedule));
}

TEST_F(BackendTest, SimpleVertexsetFilterComplete) {
    istringstream is("element Vertex end\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "func filter(v: Vertex) -> output : bool output = (age[v] > 40); end\n"
                             "func main() \n"
                             "var vertices_above_40 : vertexset{Vertex} = vertices.where(filter);"
                             "end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleFromToApplyFilter) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const old_rank : vector{Vertex}(float) = 1.0;\n"
                             "func from_filter(v: Vertex) -> output :bool output = (old_rank[v] > 40); end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (old_rank[v] < 60); end\n"
                             "func updateEdge(src : Vertex, dst : Vertex)\n"
                             "    old_rank[dst] = old_rank[src];\n"
                             "end\n"
                             "func main()\n"
                             "    edges.from(from_filter).to(to_filter).apply(updateEdge);\n"
                             "end"
    );
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleApplyReturnFrontier){
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func update (src: Vertex, dst: Vertex) -> output :bool output = true; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = "
                                    "edges.from(from_filter).to(to_filter).apply(update); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleWhileLoop){
    istringstream is("func main() while 3 < 4 print 3; end end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, VertexSetLibraryCalls){
    istringstream is("element Vertex end\n"
                            "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "func main() var frontier : vertexset{Vertex} = new vertexset{Vertex}(1); "
                             "print frontier.getVertexSetSize(); frontier.addVertex(5); print frontier.getVertexSetSize(); end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(BackendTest, SimpleApplyFromToFilterWithFromVertexsetExpression){
    istringstream is("element Vertex end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func foo(src : Vertex, dst : Vertex) -> output : bool output = true; end\n"
                             "func main() "
                             "var frontier : vertexset{Vertex} = new vertexset{Vertex}(1);"
                             "var active_vertices : vertexset{Vertex} = edges.from(frontier).to(to_filter).apply(foo); "
                             "end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleBFS){
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
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
                             "end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, SimpleEdgesetApplyPullSchedule) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                                "edges.apply(updateEdge); \n"
                             "end\n"
    );
    Schedule * schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType ::PULL};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //auto edges_apply_stmt = context_->getProgram()->elems[]
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(main_func_decl->body->stmts[0]);
    apply_stmt->stmt_label = "s1";

    graphit::Midend* me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend* be = new graphit::Backend(mir_context_);



    EXPECT_EQ (0,  be->emitCPP());

}


TEST_F(BackendTest, SimpleEdgesetApplyPushSchedule) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func updateEdge(src : Vertex, dst : Vertex) end\n"
                             "func main() \n"
                             "edges.apply(updateEdge); \n"
                             "end\n"
    );
    Schedule * schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType ::PUSH};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //auto edges_apply_stmt = context_->getProgram()->elems[]
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(main_func_decl->body->stmts[0]);
    apply_stmt->stmt_label = "s1";

    graphit::Midend* me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend* be = new graphit::Backend(mir_context_);



    EXPECT_EQ (0,  be->emitCPP());

}


TEST_F(BackendTest, SimpleEdgesetApplyNestedLabelsPullSchedule) {
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
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend* be = new graphit::Backend(mir_context_);



    EXPECT_EQ (0,  be->emitCPP());

}

TEST_F(BackendTest, SimpleForEdgesetApplyNoNestedLabelsPullSchedule) {
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
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //set the label of the edgeset apply expr stamt
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[5]);
    fir::ForStmt::Ptr for_stmt = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);

    //set the label of the for loop
    fir::ExprStmt::Ptr apply_stmt = fir::to<fir::ExprStmt>(for_stmt->body->stmts[0]);
    apply_stmt->stmt_label = "s1";

    graphit::Midend* me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend* be = new graphit::Backend(mir_context_);



    EXPECT_EQ (0,  be->emitCPP());

}


TEST_F(BackendTest, SimpleBFSPushSchedule){
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
    Schedule * schedule = new Schedule();
    ApplySchedule s1_apply_schedule = {"s1", ApplySchedule::DirectionType ::PUSH};
    auto apply_schedules = new std::map<std::string, ApplySchedule>();

    //We are constructing a nested scope label this time
    (*apply_schedules)["s1"] = s1_apply_schedule;
    schedule->apply_schedules = apply_schedules;

    fe_->parseStream(is, context_, errors_);

    //set the label of the edgeset apply expr stamt
    fir::FuncDecl::Ptr main_func_decl  =  fir::to<fir::FuncDecl>(context_->getProgram()->elems[7]);
    fir::WhileStmt::Ptr while_stmt = fir::to<fir::WhileStmt>(main_func_decl->body->stmts[2]);

    //set the label of the for loop
    fir::AssignStmt::Ptr assign_stmt = fir::to<fir::AssignStmt>(while_stmt->body->stmts[0]);
    assign_stmt->stmt_label = "s1";

    graphit::Midend* me = new graphit::Midend(context_, schedule);
    std::cout << "fir: " << std::endl;
    std::cout << *(context_->getProgram());
    std::cout << std::endl;

    me->emitMIR(mir_context_);
    graphit::Backend* be = new graphit::Backend(mir_context_);

    ofstream test_file;
    test_file.open ("../test.cpp");
    be->emitCPP(test_file);
    test_file.close();
    std::cout << exec_cmd("g++ -std=c++11 -I ../../src/runtime_lib/ ../test.cpp  -o test.o 2>&1");
    std::cout << exec_cmd("./test.o 2>&1");
    EXPECT_EQ (0,  0);
}


