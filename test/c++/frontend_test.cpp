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
                             "func main() print vector_a.sum(); end");
    EXPECT_EQ (0,  basicTest(is));
}

/**
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
                             "func srcAddOne(src : Vertex, dst : Vertex, edge : Edge) "
                             "vector_a[src] = vector_a[src] + 1; end\n"
                             "func main() edges.apply(srcAddOne); print vector_a.sum() end");
    EXPECT_EQ (0,  basicTest(is));
}
 **/