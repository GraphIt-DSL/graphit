//
// Created by Yunming Zhang on 1/20/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/frontend/error.h>

using namespace std;
using namespace graphit;


class FrontendTest : public ::testing::Test {
protected:
    virtual void SetUp(){
        context_ = new graphit::FIRContext();
        errors_ = new std::vector<ParseError>();
        fe_ = new Frontend();
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).


    }

    bool basicTest(std::istream & is){
        bool output =  fe_->parseStream(is, context_, errors_);
        //prints out the FIR, just a hack for now
        if (output == 0){
            std::cout << "fir: " << std::endl;
            std::cout << *(context_->getProgram());
            std::cout << std::endl;
        }
        return output;
    }

    std::vector<ParseError> * errors_;
    graphit::FIRContext* context_;
    Frontend * fe_;
};




//tests front end
TEST_F(FrontendTest, SimpleVarDecl ) {
    istringstream is("const a : int = 3 + 4;");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleFunctionDecl ) {
    istringstream is("func add(a : int, b: int) -> c : int  end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleFunctionDeclWithNoReturn ) {
    istringstream is("func add(a : int, b: int) end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleFunctionDecFail) {
    istringstream is("func add(a : int, b: int) ");
    EXPECT_EQ (1,  basicTest(is));
}

TEST_F(FrontendTest, SimpleFunctionDecFailNoEnd) {
    istringstream is("func add(a : int, b: int) -> c : int ");
    EXPECT_EQ (1,  basicTest(is));
}

TEST_F(FrontendTest, SimpleFunctionWithVarDecl) {
    istringstream is("func add(a : int, b: int) -> c : int var d : int = 3; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleFunctionWithAdd) {
    istringstream is("func add(a : int, b: int) -> c : int c = a + b; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, MainFunctionWithPrint) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, MainFunctionWithArgv) {
    istringstream is("func main( argv : vector[1](string)  ) print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, MainFunctionWithCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, MainFunctionWithPrintCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() print add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, ElementDecl) {
    istringstream is("element Vertex end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVecDecl) {
    istringstream is("element Vertex end\n"
                    "const vector_a : vector{Vertex}(float) = 0.0;");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVertexSet) {
    istringstream is("extern vertices : vertexset{Vertex};");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVertexSetAlloc) {
    istringstream is("const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVertexSetDeclAlloc) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVertexSetDeclAllocWithMain) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleMultiArrayAllocWithMain) {
    istringstream is("element Vertex end\n"
                             "const old_rank : vector{Vertex}(float) = 0.0;\n"
                             "const new_rank : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleEdgeSetWithMain) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "func main() print 0; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVariable){
    istringstream is("func main() var a : int = 4; print a; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleMethodCallsChaining){
    istringstream is("func main() var sum : float = vector_a.sum().foo().bar(); print sum; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleTensorRead){
    istringstream is("func main() var first_element : float = vector_a[0]; print first_element; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVectorSum){
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() var sum : float = vector_a.sum(); print sum; end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleVertexSetApply){
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() vertices.apply(addone); print vector_a.sum(); end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleVertexSetLoad){
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "func main() print 0; end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleEdgeSetApply) {
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


TEST_F(FrontendTest, SimpleForLoops) {
    istringstream is("func main() for i in 1:10; print i; end end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, VertexSetGetSize) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const size : int = vertices.size();\n"
                             "func main() print size; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, EdgeSetGetOutDegrees) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                             "func main() print out_degrees.sum(); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleFixedIterPageRank) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const old_rank : vector{Vertex}(float) = 1.0;\n"
                             "const new_rank : vector{Vertex}(float) = 0.0;\n"
                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                             "const damp : float = 0.85;\n"
                             "const beta_score : float = (1.0 - damp) / vertices.size();\n"
                             "func updateEdge(src : Vertex, dst : Vertex)\n"
                             "    new_rank[dst] = old_rank[src] / out_degrees[src];\n"
                             "end\n"
                             "func updateVertex(v : Vertex)\n"
                             "    new_rank[v] = beta_score + damp*(new_rank[v]);\n"
                             "    old_rank[v] = new_rank[v];\n"
                             "    new_rank[v] = 0.0;\n"
                             "end\n"
                             "func main()\n"
                             "    for i in 1:10\n"
                             "        edges.apply(updateEdge);\n"
                             "        vertices.apply(updateVertex);\n"
                             "    end\n"
                             "end"
                            );
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, TimerTest) {
    istringstream is("func main() \n"
                             "startTimer(); \n"
                             "var elapsed : float = stopTimer(); \n"
                             "end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVertexsetFilter) {
    istringstream is("func filter_func(v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var vertices_above_40 : vertexset{Vertex} = vertices.where(filter_func); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleVertexsetFilterComplete) {
    istringstream is("element Vertex end\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "func filter_func(v: Vertex) -> output : bool output = (age[v] > 40); end\n"
                             "func main() \n"
                             "var vertices_above_40 : vertexset{Vertex} = vertices.where(filter_func);"
                             "end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleApplyFromFilterWithBoolExpression){
    istringstream is("func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
            "func main() var active_vertices : vertexset{Vertex} = edges.from(from_filter).apply(foo); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleApplyFromToFilterWithBoolExpression){
    istringstream is("func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = edges.from(from_filter).to(to_filter).apply(foo); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleApplyReturnFrontier){
    istringstream is("func update (v: Vertex) -> output :bool output = true; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = edges.from(from_filter).to(to_filter).apply(foo); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleWhileLoop){
    istringstream is("func main() while 3 < 4 print 3; end end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, VertexSetLibraryCalls){
    istringstream is("func main() print frontier.getVertexSetSize(); frontier.addVertex(5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleApplyFromToFilterWithFromVertexsetExpression){
    istringstream is("func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = edges.from(frontier).to(to_filter).apply(foo); end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleBFS){
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

TEST_F(FrontendTest, SimpleIfElifElseStmt) {
    istringstream is("func main() var x : int = 1; if x < 1 \n"
                                                       "print \" x is less than 1\"; \n"
                                                    " elif x > 5\n"
                                                         "  print \"x is greater than 5\";\n"
                                                    "else\n"
                                                         "  print \"x is between 1 and 5\"; "
                                                   "end "
                                           "end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleWeightedEdgeSetApply) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const wt_sum : vector{Vertex}(int) = 0;\n"
                             "func sumEdgeWt(src : Vertex, dst : Vertex, weight : int) "
                             "wt_sum[dst] = wt_sum[dst] + weight; end\n"
                             "func main() edges.apply(sumEdgeWt); end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimpleBreak) {
    istringstream is("func main() "
                             "    for i in 1:10; "
                             "        if i > 1 break; end "
                             "        print i; "
                             "    end "
                             "end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, SimpleAttachLabel) {
    istringstream is("func main() "
                             "# l1 # for i in 1:2; print 4; end "
                             "end");
    EXPECT_EQ (0,  basicTest(is));

}


TEST_F(FrontendTest, SimpleAttachLabelNoSpace) {
    istringstream is("func main() "
                             "#l1# for i in 1:2; print 4; end "
                             "end");
    EXPECT_EQ (0,  basicTest(is));

}

TEST_F(FrontendTest, SimpleNestedLabelParsing) {
    istringstream is("func main() "
                             "# l1 # for i in 1:2; # s1 # print 4; end "
                             "end");



    EXPECT_EQ (0,  basicTest(is));

    fir::FuncDecl::Ptr main_func_decl = fir::to<fir::FuncDecl>(context_->getProgram()->elems[0]);
    fir::ForStmt::Ptr l1_loop = fir::to<fir::ForStmt>(main_func_decl->body->stmts[0]);
    fir::PrintStmt::Ptr s1_print = fir::to<fir::PrintStmt>(l1_loop->body->stmts[0]);

    EXPECT_EQ("l1", l1_loop->stmt_label);
    EXPECT_EQ("s1", s1_print->stmt_label);

}

TEST_F(FrontendTest, SimpleApplyWithModifiedTracking){
    istringstream is("func update (v: Vertex) -> output :bool output = true; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = edges.from(from_filter).to(to_filter).applyModified(foo, vectora); end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, SimplePlusReduce) {
    istringstream is("func add(a : int, b: int) a += b; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, UninitializedVertexProperty) {
    istringstream is("const out_degrees : vector{Vertex}(int);\n"
                             "func main() print out_degrees.sum(); end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(FrontendTest, VectorVertexProperty) {
    istringstream is("const latent_vec : vector{Vertex}(vector[20](float));\n"
                             "func main() print out_degrees.sum(); end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(FrontendTest, VertexsetFiltering) {
    istringstream is("func main() vertices.where(filter_func); end");
    EXPECT_EQ (0, basicTest(is));
}