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

TEST_F(FrontendTest, UINTVarDecl ) {
    istringstream is("const a : uint = 3 + 4;");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, UINT64VarDecl ) {
    istringstream is("const a : uint_64 = 3 + 4;");
    EXPECT_EQ (0,  basicTest(is));
}

//TEST_F(FrontendTest, UINT64WithVar ) {
//    istringstream is("var a : uint_64 = 3 + 4;");
//    EXPECT_EQ (0,  basicTest(is));
//}
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

TEST_F(FrontendTest, SimpleIntersectionOperator) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                     "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                     "const inter: uint_64 = intersection(vertices1, vertices2, 0, 0);\n");
    EXPECT_EQ (0, basicTest(is));

}

TEST_F(FrontendTest, SimpleIntersectionOperatorWithOptional) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const vertices1 : vertexset{Vertex} = edges.getVertices();\n"
                     "const vertices2 : vertexset{Vertex} = edges.getVertices();\n"
                     "const inter: uint_64 = intersection(vertices1, vertices2, 0, 0, 5);\n");
    EXPECT_EQ (0, basicTest(is));

}

TEST_F(FrontendTest, MainFunctionWithPrint) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, MainFunctionWithArgv) {
    istringstream is("func main( argv : vector[1](string)  ) print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, ExportFunction) {
    istringstream is("export func export_func() print 4; end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(FrontendTest, EdgeSetExportFuncVectorInit) {
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
                     "      vector_a = new vector{Vertex}(float)(); "
                     "      vertices.apply(update_vector); "
                     " end");
    EXPECT_EQ (0, basicTest(is));
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

TEST_F(FrontendTest, SimpleIntersectNeighborOperator) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex);\n"
                     "const src : int = 0;\n"
                     "const dest: int = 0;\n"
                     "const inter : uint_64 = intersectNeighbor(edges, src, dest);\n");
    EXPECT_EQ (0, basicTest(is));


}

TEST_F(FrontendTest, SimpleIntersectNeighborOperatorInsideMain) {
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



TEST_F(FrontendTest, PriorityQueueDeclaration) {
    istringstream is("element Vertex end func main() var pq: priority_queue{Vertex}(int); end");
    EXPECT_EQ(0, basicTest(is));
}

TEST_F(FrontendTest, PriorityQueueAllocation) {
    istringstream is("element Vertex end func udf() end func main() var pq: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(false, false, udf, 1, 2, false, -1); end");
    EXPECT_EQ(0, basicTest(is));
}


TEST_F(FrontendTest, GlobalPriorityQueueAllocation) {
    istringstream is("element Vertex end "
                     "const pq: priority_queue{Vertex}(int);"
                     "func udf() end "
                     "func main() "
                     "      pq = new priority_queue{Vertex}(int)(false, false, udf, 1, 2, false, -1); "
                     "end");
    EXPECT_EQ(0, basicTest(is));
}


TEST_F(FrontendTest, PriorityQueueMultipleAllocation) {
    istringstream is("element Vertex end\n"
		     "func udf() end\n"
                     "func main()\n"
                     "    var pq: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(false, false, udf, 1, 2, false, -1);\n"
                     "    var pq2: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(true, true, udf, 0, 0, false, 1024);\n"
                     "end");
    EXPECT_EQ(0, basicTest(is));
}

TEST_F(FrontendTest, PriorityQueueWithDelete) {
    istringstream is("element Vertex end\n"
		     "func udf() end\n"
                     "func main()\n"
                     "    var pq: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(false, false, udf, 1, 2, false, -1);\n"
		                  "delete pq;\n"
                     "end");
    EXPECT_EQ(0, basicTest(is));
}


TEST_F(FrontendTest, PriorityQueueWithVector) { 
    istringstream is("element Vertex end\n"
		     "const array: vector{Vertex}(int) = 0;\n"
                     "func main()\n"
                     "    var pq: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(false, false, array, 1, 2, false, -1);\n"
                     "end");
    EXPECT_EQ(0, basicTest(is));
}


TEST_F(FrontendTest, PriorityQueueLibraryCall) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges: edgeset{Edge}(Vertex, Vertex, int);\n"
                     "const array: vector{Vertex}(int) = 0;\n"
                     "func main()\n"
                     "    var pq: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(false, false, array, 1, 2, false, -1);\n"
                     "    var frontier: vertexset{Vertex} = pq.get_min_bucket();\n"
                     "end");
    EXPECT_EQ(0, basicTest(is));
}

TEST_F(FrontendTest, PriorityQueueApplyUpdatePriority) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges: edgeset{Edge}(Vertex, Vertex, int);\n"
		             "const array: vector{Vertex}(int) = 0;\n"
                     "func udf(src: Vertex, dst: Vertex) end\n"
                     "func main()\n"
                     "    var pq: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(false, false, array, 1, 2, false, -1);\n"
                     "    var frontier: vertexset{Vertex} = pq.get_min_bucket();\n"
                     "    edges.from(frontier).applyUpdatePriority(udf);\n"
                     "end");
    EXPECT_EQ(0, basicTest(is));
}

TEST_F(FrontendTest, PriorityQueueApplyUpdatePriorityExtern) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges: edgeset{Edge}(Vertex, Vertex, int);\n"
		            "const array: vector{Vertex}(int) = 0;\n"
                     "extern func udf(src: Vertex, dst: Vertex);\n"
                     "func main()\n"
                     "    var pq: priority_queue{Vertex}(int) = new priority_queue{Vertex}(int)(false, false, array, 1, 2, false, -1);\n"
                     "    var frontier: vertexset{Vertex} = pq.get_min_bucket();\n"
                     "    frontier.applyUpdatePriorityExtern(udf);\n"
                     "end");
    EXPECT_EQ(0, basicTest(is));
}

TEST_F(FrontendTest, DeltaStepping) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.wel\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const dist : vector{Vertex}(int) = 2147483647; %should be INT_MAX\n"
                     "const pq: priority_queue{Vertex}(int);"

                     "func updateEdge(src : Vertex, dst : Vertex, weight : int) \n"
                            "var new_dist : int = dist[src] + weight; "
                            "pq.updatePriorityMin(dst, new_dist, dist[dst]); "
                     "end\n"
                     "func main() "
                     "  var start_vertex : Vertex = 1;"
                     "  pq = new priority_queue{Vertex}(int)(false, false, array, 1, 2, false, -1);"
                     "  while (not pq.finished()) "
                       "    var frontier : vertexsubset = pq.get_current_priority_nodes(); \n"
                         "  edges.from(frontier).applyUpdatePriority(updateEdge);  \n"
                         "  delete frontier; "
                         "end\n"
                     "end");
    EXPECT_EQ (0,  basicTest(is));

}


TEST_F(FrontendTest, PointToPointShortestPath) {
    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex,Vertex) = load (\"../test/graphs/test.wel\");\n"
                     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
                     "const dist : vector{Vertex}(int) = 2147483647; %should be INT_MAX\n"
                     "const pq: priority_queue{Vertex}(int);"
                     "func updateEdge(src : Vertex, dst : Vertex, weight : int) \n"
                     "  var new_dist : int = dist[src] + weight; "
                     "  pq.updatePriorityMin(dst, new_dist, dist[dst]); "
                     "end\n"
                     "func main() "
                     "  var start_vertex : Vertex = 1;"
                     "  var end_vertex : Vertex = 10; "
                     "  pq = new priority_queue{Vertex}(int)(false, false, array, 1, 2, false, -1);"
                     "  while (not pq.finishedNode(end_vertex)) "
                     "    var frontier : vertexsubset = pq.get_current_priority_nodes(); \n"
                     "    edges.from(frontier).applyUpdatePriority(updateEdge);  \n"
                     "    delete frontier; "
                     "  end\n"
                     "end");
    EXPECT_EQ (0,  basicTest(is));
}


TEST_F(FrontendTest, KCoreFrontendTest) {
    istringstream is("element Vertex end\n"
		     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
		     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
		     "const D: vector{Vertex}(int) = edges.getOutDegrees();\n"
		     "const pq: priority_queue{Vertex}(int);\n"
                     "func apply_f(src: Vertex, dst: Vertex)\n"
                     "    var k: int = pq.get_current_priority();\n"
                     "    pq.updatePrioritySum(dst, -1, k);\n"
                     "end \n"
		     "func main()\n"
                     "    pq = new priority_queue{Vertex}(int)(false, false, D, 1, 2, false, -1);\n"
                     "    var finished: int = 0; \n"
                     "    while (finished != vertices.size()) \n"
                     "        var frontier: vertexset{Vertex} = pq.get_min_bucket();\n"
                     "        finished += frontier.size();\n"
                     "        edges.from(frontier).applyUpdatePriority(apply_f);\n"
                     "        delete frontier;\n"
                     "    end\n"
                     "    delete pq;\n"
                     "end\n"
    );
    EXPECT_EQ(0, basicTest(is));
}


TEST_F(FrontendTest, SetCoverFrontendTest) {
    istringstream is("element Vertex end\n"
		     "element Edge end\n"
		     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
		     "const vertices : vertexset{Vertex} = edges.getVertices();\n"
		     "const degrees: vector{Vertex}(int) = edges.getOutDegrees();\n"
		     "const D: vector{Vertex}(int);\n"
		     "const pq: priority_queue{Vertex}(int);\n"
             "const epsilon: double = 0.01;\n"
		     "const x: double = 1.0/log(1.0 + epsilon);\n"
		     "func init_udf(v: Vertex) \n"
	             "var deg: int = degrees[v];\n"
                     "    if deg == 0\n"
                     "        D[v] = 0;\n"
                     "    else\n"
                     "        D[v] = floor(x*log(to_double(deg)));\n"
                     "    end\n"
		     "end\n"
		     "extern func extern_function(active: vertexset{Vertex}) -> output: vertexset{Vertex};\n"
		     "func main() \n"
                     "    vertices.apply(init_udf);\n"
	 	     "            pq = new priority_queue{Vertex}(int)(false, false, D, 1, 2, false, -1);\n"
                     "     while (1) \n"
                     "        var frontier: vertexset{Vertex} = pq.get_min_bucket();\n"
                     "        if frontier.is_null()\n"
		     "            break;\n"
                     "        end\n"
                     "        frontier.applyUpdatePriorityExtern(extern_function);\n"
                     "        delete frontier;\n"
                     "    end\n"
                     "end\n"
    );
    EXPECT_EQ(0, basicTest(is));
}

TEST_F(FrontendTest, FunctorOneStateDeclTest) {

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


TEST_F(FrontendTest, FunctorOneStateTest) {

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

TEST_F(FrontendTest, FunctorOneStateEdgesetTest) {

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



TEST_F(FrontendTest, FunctorMultipleStatesTest) {

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

TEST_F(FrontendTest, LocalVectorInitTest) {

    istringstream is("element Vertex end\n"
                     "element Edge end\n"
                     "const edges : edgeset{Edge}(Vertex, Vertex) = load (\"test.el\");\n"
                     "func main()\n"
                     "    var simpleArray: vector{Vertex}(int) = 0;\n"
                     "end\n"

    );
    EXPECT_EQ(0, basicTest(is));

}


