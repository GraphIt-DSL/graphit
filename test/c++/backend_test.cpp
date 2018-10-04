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
        return be->emit();
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
