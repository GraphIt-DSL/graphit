//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/midend/midend.h>
#include <graphit/frontend/error.h>

using namespace std;
using namespace graphit;


class MidendTest : public ::testing::Test {
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
        return me->emitMIR(mir_context_);
    }


    std::vector<ParseError> * errors_;
    graphit::FIRContext* context_;
    Frontend * fe_;
    graphit::MIRContext* mir_context_;
};


//tests mid end
TEST_F(MidendTest, SimpleVarDecl ) {
    istringstream is("const a : int = 3 + 4;");
    EXPECT_EQ (0 , basicTest(is));
}


TEST_F(MidendTest, UINTDecl ) {
    istringstream is("const a : uint = 3 + 4;");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, UINT64Decl ) {
    istringstream is("const a : uint_64 = 3 + 4;");
    EXPECT_EQ (0 , basicTest(is));
}
//tests mid end
TEST_F(MidendTest, SimpleFunctionDecl) {
    istringstream is("func add(a : int, b: int) -> c : int  end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, SimpleFunctionDeclWithNoReturn) {
    istringstream is("func add(a : int, b: int)  end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, SimpleFunctionWithVarDecl) {
    istringstream is("func add(a : int, b: int) -> c : int var d : int = 3; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, SimpleFunctionWithAdd) {
    istringstream is("func add(a : int, b: int) -> c : int c = a + b; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, MainFunctionWithPrint) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, MainFunctionWithCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, MainFunctionWithPrintCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() print add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, ElementDecl) {
    istringstream is("element Vertex end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, SimpleVertexSetDeclAlloc) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, SimpleVertexSetDeclAllocWithMain) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() print 4; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(MidendTest, SimpleIntersectionOperator) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                     "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                     "const inter: uint_64 = intersection(vertices1, vertices2, 0, 0);\n");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(MidendTest, FunctorOneStateDeclTest) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const simpleArray: vector{Vertex}(int) = 0;\n"
                     "func addStuff[a: int](v: Vertex)\n"
                     "    simpleArray[v] += a;\n"
                     "end\n"
                     "func main()\n"
                     "    var test: int = 5;\n"
                     "end\n"

    );
    EXPECT_EQ(0, basicTest(is));

}


TEST_F(MidendTest, FunctorOneStateTest) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const simpleArray: vector{Vertex}(int) = 0;\n"
                     "func addStuff[a: int](v: Vertex)\n"
                     "    simpleArray[v] += a;\n"
                     "end\n"
                     "func main()\n"
                     "    var test: int = 5;\n"
                     "    vertices.apply(addStuff[test]);\n"
                     "    addStuff[test](0); \n"
                     "end\n"

    );
    EXPECT_EQ(0, basicTest(is));

}

TEST_F(MidendTest, FunctorOneStateEdgesetTest) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const simpleArray: vector{Vertex}(int) = 0;\n"
                     "func addStuff[a: int](src : Vertex, dst : Vertex)\n"
                     "    simpleArray[src] += a;\n"
                     "    simpleArray[dst] += a;\n"
                     "end\n"
                     "func main()\n"
                     "    var test: int = 5;\n"
                     "    edges.apply(addStuff[test]);\n"
                     "    addStuff[test](0); \n"
                     "end\n"

    );
    EXPECT_EQ(0, basicTest(is));

}



TEST_F(MidendTest, FunctorMultipleStatesTest) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const simpleArray: vector{Vertex}(int) = 0;\n"
                     "func addStuff[a: int, b: float](v: Vertex)\n"
                     "    print b;\n"
                     "    simpleArray[v] += a;\n"
                     "end\n"
                     "func main()\n"
                     "    var test: int = 5;\n"
                     "    var test_v2: float = 5.0;\n"
                     "    vertices.apply(addStuff[test, test_v2]);\n"
                     "end\n"

    );
    EXPECT_EQ(0, basicTest(is));

}