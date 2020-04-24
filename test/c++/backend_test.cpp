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

    std::vector<ParseError> *errors_;
    graphit::FIRContext *context_;
    Frontend *fe_;
    graphit::MIRContext *mir_context_;
};

//tests back end
TEST_F(BackendTest, SimpleVarDecl) {
    istringstream is("const a : int = 3 + 4;");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, UINTGlobalDecl) {
    istringstream is("const a : uint = 3 + 4;");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, UINT64GlobalDecl) {
    istringstream is("const a : uint_64 = 3 + 4;");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(BackendTest, UINTGlobalLocalIncr) {
    istringstream is("const a : uint = 0;\n"
                     "func main() a += 1; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, UINT64GlobalLocalIncr) {
    istringstream is("const a : uint_64 = 0;\n"
                     "func main() a += 1; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, UINTReassign) {
    istringstream is("const a : uint;\n"
                     "func main() a = 1; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, UINT64Reassign) {
    istringstream is("const a : uint_64;\n"
                     "func main() a = 1; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, UINTLocalDec) {
    istringstream is("func main() const a : uint = 1; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, UINT64LocalDec) {
    istringstream is("func main() const a : uint_64 = 1; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleDoubleVarDecl) {
    istringstream is("const a : double = 3; \n func main()  end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleFloatVarDecl) {
    istringstream is("const a : float = 3.0; \n func main()  end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionDecl) {
    istringstream is("func add(a : int, b: int) -> c : int  end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionDeclWithNoReturn) {
    istringstream is("func add(a : int, b: int)  end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionWithVarDecl) {
    istringstream is("func add(a : int, b: int) -> c : int var d : int = 3; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleFunctionWithAdd) {
    istringstream is("func add(a : int, b: int) -> c : int c = a + b; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, MainFunctionWithPrint) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, MainFunctionWithCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() add(4, 5); end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, MainFunctionWithPrintCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() print add(4, 5); end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, ElementDecl) {
    istringstream is("element Vertex end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleVertexSetDeclAlloc) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleUINTVector) {
    istringstream is("element Vertex end\n"
                     "const vector_a : vector{Vertex}(uint) = 0;\n");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleUINT64Vector) {
    istringstream is("element Vertex end\n"
                     "const vector_a : vector{Vertex}(uint_64) = 0;\n");
    EXPECT_EQ (0, basicTest(is));
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
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleEdgeSetWithMain) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "func main() print 0; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleVariable) {
    istringstream is("func main() var a : int = 4; print a; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleVectorSum) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() \n"
                             "     var summation : float = vector_a.sum(); \n"
                             "     print summation; \n"
                             "end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleVertexSetApply) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func addone(v : Vertex) vector_a[v] = vector_a[v] + 1; end \n"
                             "func main() "
                             "      vertices.apply(addone); "
                             "      print vector_a.sum(); "
                             "end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleEdgeSetLoad) {
    istringstream is("element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "func main() print 0; end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleVertexSetLoad) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vector_a : vector{Vertex}(float) = 1.0;\n"
                             "func main() print 0; end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, ExportSimpleVertexSetLoad) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "export func process() print 0; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, ExportReturnConstantSizeVector) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "export func process() -> output : vector[10](int) print 0; end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, ExportSimpleVertexSetLoadInFunction) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices : vertexset{Vertex};\n"
                     "export func process() "
                     "      edges = load (\"test.el\");"
                     "      vertices = edges.getVertices();"
                     " end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, EdgeSetExportFunc) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices : vertexset{Vertex};\n"
                     "export func process(input_edges : edgeset{Edge}(Vertex,Vertex)) "
                     "      edges = input_edges;"
                     "      vertices = edges.getVertices();"
                     " end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, EdgeSetExportFuncVectorInit) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices : vertexset{Vertex};\n"
                     "const vector_a : vector{Vertex}(float);\n"
                     "func update_vector_a (v : Vertex ) "
                     "  vector_a[v] = 0; "
                     "end \n"
                     "export func process(input_edges : edgeset{Edge}(Vertex,Vertex)) "
                     "      edges = input_edges;"
                     "      vertices = edges.getVertices();"
                     "      vector_a = new vector{Vertex}(float)();"
                     "      vertices.apply(update_vector);"
                     " end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, EdgeSetExportFuncVectorInitWithReturn) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices : vertexset{Vertex};\n"
                     "const vector_a : vector{Vertex}(float);\n"
                     "func update_vector_a (v : Vertex ) "
                     "  vector_a[v] = 0; "
                     "end \n"
                     "export func process(input_edges : edgeset{Edge}(Vertex,Vertex)) -> output : vector{Vertex}(float) "
                     "      edges = input_edges;"
                     "      vertices = edges.getVertices();"
                     "      vector_a = new vector{Vertex}(float)();"
                     "      vertices.apply(update_vector); "
                     "      output = vector_a;"
                     " end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, EdgeSetExportFuncVectorInputWithReturn) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices : vertexset{Vertex};\n"
                     "const vector_a : vector{Vertex}(float);\n"
                     "func update_vector_a (v : Vertex ) "
                     "  vector_a[v] = 0; "
                     "end \n"
                     "export func process(input_edges : edgeset{Edge}(Vertex,Vertex), input_vector : vector{Vertex}(float)) -> output : vector{Vertex}(float) "
                     "      edges = input_edges;"
                     "      vertices = edges.getVertices();"
                     "      vector_a = input_vector;"
                     "      vertices.apply(update_vector); "
                     "      output = vector_a;"
                     " end");
    EXPECT_EQ (0, basicTest(is));
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
    EXPECT_EQ (0, basicTest(is));
}



TEST_F(BackendTest, SimpleEdgeSetTranspose) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "func srcAddOne(src : Vertex, dst : Vertex) "
                             "      vector_a[src] = vector_a[src] + 1; end\n"
                             "func main() "
                             "      var transposed_edges : edgeset{Edge}(Vertex, Vertex) = edges.transpose(); \n"
                             "      transposed_edges.apply(srcAddOne); \n"
                             " end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleForLoops) {
    istringstream is("func main() for i in 1:10; print i; end end");
    EXPECT_EQ (0, basicTest(is));
}



TEST_F(BackendTest, SimpleIntegerList) {
    istringstream is("func main() var int_list : list{int} = new list{int}(); int_list.append(1); end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleIntegerListPop) {
    istringstream is("func main() var int_list : list{int} = new list{int}(); "
                             "         int_list.append(1); "
                             "         var one : int = int_list.pop();"
                             "end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleVertexSetList) {
    istringstream is("element Vertex end\n"
                             "func main() "
                             "  var frontier : vertexset{Vertex} = new vertexset{Vertex}(0); "
                             "  var vertexset_list : list{vertexset{Vertex}} = new list{vertexset{Vertex}}(); "
                             "  vertexset_list.append(frontier);"
                             " end");
    EXPECT_EQ (0, basicTest(is));
}



TEST_F(BackendTest, SimpleVertexSetListPop) {
    istringstream is("element Vertex end\n"
                             "func main() "
                             "  var frontier : vertexset{Vertex} = new vertexset{Vertex}(0); "
                             "  var vertexset_list : list{vertexset{Vertex}} = new list{vertexset{Vertex}}(); "
                             "  vertexset_list.append(frontier);"
                             "  var pop_frontier : vertexset{Vertex} = vertexset_list.pop(); "
                             " end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, VertexSetGetSize) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const size : int = vertices.size();\n"
                             "func main() print size; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, EdgeSetGetOutDegrees) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                             "func main() print out_degrees.sum(); end");
    EXPECT_EQ (0, basicTest(is));
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
                             "    new_rank[dst] += old_rank[src] / out_degrees[src];\n"
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
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleVertexsetFilterComplete) {
    istringstream is("element Vertex end\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "func to_filter(v: Vertex) -> output : bool output = (age[v] > 40); end\n"
                             "func main() \n"
                             "var vertices_above_40 : vertexset{Vertex} = vertices.filter(to_filter);"
                             "end");
    EXPECT_EQ (0, basicTest(is));
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
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleApplyReturnFrontier) {
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
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleWhileLoop) {
    istringstream is("func main() while 3 < 4 print 3; end end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, VertexSetLibraryCalls) {
    istringstream is("element Vertex end\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "func main() var frontier : vertexset{Vertex} = new vertexset{Vertex}(1); "
                             "print frontier.getVertexSetSize(); frontier.addVertex(5); print frontier.getVertexSetSize(); end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleApplyFromToFilterWithFromVertexsetExpression) {
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
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleApplyDstFilterWithFromVertexsetExpression) {
    istringstream is("element Vertex end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func foo(src : Vertex, dst : Vertex) -> output : bool output = true; end\n"
                             "func main() "
                             "var frontier : vertexset{Vertex} = new vertexset{Vertex}(1);"
                             "var active_vertices : vertexset{Vertex} = edges.from(frontier).dstFilter(to_filter).apply(foo); "
                             "end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleBFS) {
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
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleIfElifElseStmt) {
    istringstream is("func main() var x : int = 1; if x < 1 \n"
                             "print \" x is less than 1\"; \n"
                             " elif x > 5\n"
                             "  print \"x is greater than 5\";\n"
                             "else\n"
                             "  print \"x is between 1 and 5\"; "
                             "end "
                             "end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleWeightedEdgeSetApply) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const wt_sum : vector{Vertex}(int) = 0;\n"
                             "func sumEdgeWt(src : Vertex, dst : Vertex, weight : int) "
                             "wt_sum[dst] = wt_sum[dst] + weight; end\n"
                             "func main() edges.apply(sumEdgeWt); end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleSSSP) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (\"../test/graphs/test.wel\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const SP : vector{Vertex}(int) = 2147483647; %should be INT_MAX \n"
                             "func updateEdge(src : Vertex, dst : Vertex, weight : int) -> output : bool\n"
                             "if SP[dst] > SP[src] + weight \n"
                             "SP[dst] = SP[src] + weight;\n"
                             "output = true;\n"
                             "else\n"
                             "output = false;\n"
                             "end\n"
                             "end\n"
                             "func main() \n"
                             "var n : int = edges.getVertices();\n"
                             "var frontier : vertexset{Vertex} = new vertexset{Vertex}(0);\n"
                             "frontier.addVertex(1); %add source vertex \n"
                             "SP[1] = 0;\n"
                             "var rounds : int = 0;\n"
                             "while (frontier.getVertexSetSize() != 0)\n"
                             "for i in 1:n;\n"
                             "print SP[i];\n"
                             "end \n"
                             "print \"number of active_vertices: \";\n"
                             "print frontier.getVertexSetSize();\n"
                             "frontier = edges.from(frontier).apply(updateEdge);\n"
                             "rounds = rounds + 1;\n"
                             "if rounds == n\n"
                             "print \"negative cycle\";\n"
                             "end\n"
                             " end\n"

                             "print rounds;\n"
                             "var elapsed_time : float = stopTimer();\n"
                             "print \"elapsed time: \";\n"
                             "print elapsed_time;\n"
                             "end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleBreak) {
    istringstream is("func main() "
                             "    for i in 1:10; "
                             "        if i > 1 break; end "
                             "        print i; "
                             "    end "
                             "end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleAssignReturnFrontier) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func update (src: Vertex, dst: Vertex) age[dst] = 0; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = "
                             "edges.from(from_filter).to(to_filter).applyModified(update, age); end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SimpleAssignReturnFrontierNewAPI) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func update (src: Vertex, dst: Vertex) age[dst] = 0; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = "
                             "edges.from(from_filter).to(to_filter).applyModified(update, age, false); end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, SrcFilterDstFilterApply) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func update (src: Vertex, dst: Vertex) age[dst] = 0; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = "
                             "edges.srcFilter(from_filter).dstFilter(to_filter).applyModified(update, age); end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimplePlusReduce) {
    istringstream is("func reduce_test(a : int, b: int) a += b; end");

    EXPECT_EQ (0, basicTest(is));
    EXPECT_EQ (mir_context_->getFunction("reduce_test")->result.isInitialized(), false);

}


TEST_F(BackendTest, SimpleMinReduce) {
    istringstream is("func reduce_test(a : int, b: int) a min= b; end");
    EXPECT_EQ (0, basicTest(is));
    EXPECT_EQ (mir_context_->getFunction("reduce_test")->result.isInitialized(), false);

}


TEST_F(BackendTest, SimpleAsyncMinReduce) {
    istringstream is("func reduce_test(a : int, b: int) a asyncMin= b; end");
    EXPECT_EQ (0, basicTest(is));
    EXPECT_EQ (mir_context_->getFunction("reduce_test")->result.isInitialized(), false);

}

TEST_F(BackendTest, SimpleMaxReduce) {
    istringstream is("func reduce_test(a : int, b: int) a max= b; end");
    EXPECT_EQ (0, basicTest(is));
    EXPECT_EQ (mir_context_->getFunction("reduce_test")->result.isInitialized(), false);

}


TEST_F(BackendTest, SimpleAsyncMaxReduce) {
    istringstream is("func reduce_test(a : int, b: int) a asyncMax= b; end");
    EXPECT_EQ (0, basicTest(is));
    EXPECT_EQ (mir_context_->getFunction("reduce_test")->result.isInitialized(), false);
}

TEST_F(BackendTest, SimpleMinReduceReturnFrontier) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func update (src: Vertex, dst: Vertex) age[dst] min= 1; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() "
                             "  var active_vertices : vertexset{Vertex} = "
                             "      edges.from(from_filter).to(to_filter).applyModified(update,age); "
                             "end");
    EXPECT_EQ (0, basicTest(is));
    EXPECT_TRUE (mir_context_->getFunction("update")->result.getName() != "");
}

TEST_F(BackendTest, MinReduceReturnFrontier) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const age : vector{Vertex}(int) = 0;\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "func update (src: Vertex, dst: Vertex) age[dst] min= age[src]; end\n"
                             "func to_filter (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                             "func from_filter (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                             "func main() var active_vertices : vertexset{Vertex} = "
                             "edges.from(from_filter).to(to_filter).applyModified(update, age); end");
    EXPECT_EQ (0, basicTest(is));
    EXPECT_TRUE (mir_context_->getFunction("update")->result.getName() != "");
    EXPECT_EQ (mir_context_->getFunction("update")->result.isInitialized(), true);

}

TEST_F(BackendTest, ReadCmdLineArgs) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[0]);\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const out_degrees : vector{Vertex}(int) = edges.getOutDegrees();\n"
                             "func main() print out_degrees.sum(); end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, UninitializedVertexProperty) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[0]);\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const out_degrees : vector{Vertex}(int);\n"
                             "func main()  end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, RandomNghTest) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[0]);\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "func main() var v : Vertex = getRandomOutNgh(edges, 0);  end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SerialMinimumSpanningTreeTest) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (argv[0]);\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const parents : vector{Vertex}(int);"
                     "func main() parents = serialMinimumSpanningTree(edges, 0);  end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, VectorVertexProperty) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[0]);\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const latent_vec : vector{Vertex}(vector[20](float));\n"
                             "func main()  end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, VectorVertexPropertyAccess) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[0]);\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const latent_vec : vector{Vertex}(vector[20](float));\n"
                             "func main() var f1 : float = latent_vec[0][0]; end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, FabsWithVectorRead) {
    istringstream is("element Vertex end\n"
                             "element Edge end\n"
                             "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                             "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                             "const vec : vector{Vertex}(float);\n"
                             "func updateVertex(v : Vertex) var a : float = fabs(vec[v]); end\n"
                             "func main() vertices.apply(updateVertex); end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, CF) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (argv[1]);\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const latent_vec : vector{Vertex}(vector[1](float));\n"
                     "const error_vec : vector{Vertex}(vector[1](float));\n"
                     "const step : float = 0.35; %for testing\n"
                     "const lambda : float = 0.001;\n"
                     "const K : int = 1;\n"
                     "\n"
                     "func updateEdge (src : Vertex, dst : Vertex, rating : int)\n"
                     "    var estimate : float = 0;\n"
                     "    for i in 0:K\n"
                     "        estimate  += latent_vec[src][i] * latent_vec[dst][i];\n"
                     "    end\n"
                     "    var err : float =  rating - estimate;\n"
                     "    for i in 0:K\n"
                     "        error_vec[dst][i] += latent_vec[src][i]*err;\n"
                     "    end\n"
                     "end\n"
                     "\n"
                     "func updateVertex (v : Vertex)\n"
                     "     for i in 0:K\n"
                     "        latent_vec[v][i] += step*(-lambda*latent_vec[v][i] + error_vec[v][i]);\n"
                     "        error_vec[v][i] = 0;\n"
                     "     end\n"
                     "end\n"
                     "\n"
                     "func initVertex (v : Vertex)\n"
                     "    for i in 0:K\n"
                     "        latent_vec[v][i] = 0.5;\n"
                     "        error_vec[v][i] = 0;\n"
                     "    end\n"
                     "end\n"
                     "\n"
                     "func main()\n"
                     "    vertices.apply(initVertex);\n"
                     "    for i in 0:5\n"
                     "        #s1# edges.apply(updateEdge);\n"
                     "        vertices.apply(updateVertex);\n"
                     "    end\n"
                     "\n"
                     "    var sum : float = 0;\n"
                     "    for i in 0:edges.getVertices()\n"
                     "        for j in 0:K\n"
                     "            sum += latent_vec[i][j];\n"
                     "        end\n"
                     "    end\n"
                     "\n"
                     "    print sum;\n"
                     "\n"
                     "end\n"
                     );
    EXPECT_EQ (0, basicTest(is));
}



TEST_F(BackendTest, vertexsetApplyExtern) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const vec : vector{Vertex}(float);\n"
                     "extern func UDF_updateVertex(v : Vertex);\n"
                     "func main() vertices.apply(UDF_updateVertex); end");
    EXPECT_EQ (0, basicTest(is));
}



TEST_F(BackendTest, edgesetApplyExtern) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const vec : vector{Vertex}(float);\n"
                     "extern func UDF_updateEdge(src : Vertex, dst : Vertex);\n"
                     "func main() edges.apply(UDF_updateEdge); end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, vectorPerVertexTestNoConstDef) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[1]);\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "export func foo(v: vector{Vertex}(vector[20](int)))\n"
                     "      for i in 0:20\n"
                     "            print v[i][i];\n"
                     "      end\n"
                     "end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, vectorPerVertexTestWithConstDef) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[1]);\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const v: vector{Vertex}(vector[20](int));\n"
                     "export func foo(v: vector{Vertex}(vector[20](int)))\n"
                     "      for i in 0:20\n"
                     "            print v[i][i];\n"
                     "      end\n"
                     "end");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, newVertexsetFilterTest) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[1]);\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "func filter_func(v : Vertex) -> output : bool \n"
                     "  output = true; \n"
                     "end\n"
                     "export func foo()\n"
                     "  var vset : vertexset{Vertex} = new vertexset{Vertex}(0); \n"
                     "  vset = vset.filter(filter_func); \n"
                     "end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, GlobalConstantSizeVectorTest) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (argv[1]);\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const float_vector : vector{Vertex}(float) = 0.0;\n"
                     "const constant_vector : vector[100](float); \n "
                     "export func foo(input_vector : vector[100](float))\n"
                     "  constant_vector = input_vector;"
                     "end");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleIntersectionOperator) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                     "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                     "const inter: uint_64 = intersection(vertices1, vertices2, 0, vertices2);\n");
    EXPECT_EQ (0, basicTest(is));

}

TEST_F(BackendTest, SimpleIntersectionOperatorInsideMain) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                     "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                     "func main()\n"
                     "     var inter : uint_64 = intersection(vertices1, vertices2, 0, 0);\n"
                     "end\n");
    EXPECT_EQ (0, basicTest(is));

}

TEST_F(BackendTest, SimpleIntersectionOperatorWithOptional) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                     "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                     "const inter: uint_64 = intersection(vertices1, vertices2, 0, 0, 5);\n");
    EXPECT_EQ (0, basicTest(is));

}

TEST_F(BackendTest, SimpleIntersectNeighborOperator) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const src : int = 0;\n"
                     "const dest: int = 0;\n"
                     "const inter : uint_64 = intersectNeighbor(edges, src, dest);\n");
    EXPECT_EQ (0, basicTest(is));
}

TEST_F(BackendTest, SimpleIntersectNeighborOperatorInsideMain) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const src : int = 0;\n"
                     "const dest: int = 0;\n"
                     "func main()\n"
                     "     var inter : uint_64 = intersectNeighbor(edges, src, dest);\n"
                     "end\n");
    EXPECT_EQ (0, basicTest(is));
}


TEST_F(BackendTest, VectorInitWithoutVertex) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex, int) = load (argv[1]);\n"
                     "% const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const vertexArray: vector{Vertex}(int) = 0;\n"
                     "func main()\n"
                     "     print vertexArray;\n"
                     "end");
    EXPECT_EQ (0, basicTest(is));

}

TEST_F(BackendTest, FunctorOneStateTest) {

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
                     "end\n"

    );
    EXPECT_EQ(0, basicTest(is));

}


TEST_F(BackendTest, FunctorMultipleStatesTest) {

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


TEST_F(BackendTest, FunctorEdgesetApplyModified) {

    istringstream is("element Vertex end\n"
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

    EXPECT_EQ(0, basicTest(is));

}

TEST_F(BackendTest, ExportLocalVectorWithNew) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const simpleArray: vector{Vertex}(vector[1](float));\n"
                     "const simpleArray2: vector{Vertex}(vector[1](int));\n"
                     "const K: int = 1;\n"
                     "func initVertex (v : Vertex)\n"
                     "      for i in 0:K\n"
                     "          simpleArray[v][i] = 0.5;\n"
                     "          simpleArray2[v][i] = 0;\n"
                     "      end\n"
                     "end\n"
                     "export func export_func() -> output : vector{Vertex}(vector[1](float))\n"
                     "    simpleArray = new vector{Vertex}(vector[1](float))();\n"
                     "    simpleArray2 = new vector{Vertex}(vector[1](int))();\n"
                     "    vertices.apply(initVertex);\n"
                     "    output = simpleArray;\n"
                     "end\n");

    EXPECT_EQ(0, basicTest(is));

}


TEST_F(BackendTest, SrcFilterDstFilterApplyFunctor) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "func update[age: vector{Vertex}(int)] (src: Vertex, dst: Vertex) age[dst] = 0; end\n"
                     "func to_filter[age: vector{Vertex}(int)] (v: Vertex) -> output :bool output = (age[v] < 60); end\n"
                     "func from_filter[age: vector{Vertex}(int)] (v: Vertex) -> output :bool output = (age[v] > 40); end\n"
                     "func main()\n"
                     "var age: vector{Vertex}(int) = 0;\n"
                     "var active_vertices : vertexset{Vertex} = "
                     "edges.srcFilter(from_filter[age]).dstFilter(to_filter[age]).apply(update[age]);\n"
                     "end\n");
    EXPECT_EQ (0, basicTest(is));
}

//TODO: should be supported soon
TEST_F(BackendTest, LocalVector) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "func main()"
                     "    var simpleArray: vector{Vertex}(vector[1](int)) = 0;\n"
                     "end\n");

    EXPECT_EQ(0, basicTest(is));
}

