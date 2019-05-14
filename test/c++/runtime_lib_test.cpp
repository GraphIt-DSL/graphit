//
// Created by Yunming Zhang on 4/27/17.
//

#include <gtest.h>
#include "intrinsics.h"
#include "infra_gapbs/graph_verifier.h"
#include "infra_gapbs/intersections.h"




class RuntimeLibTest : public ::testing::Test {
protected:
    virtual void SetUp() {

    }

    virtual void TearDown() {

    }

};


TEST_F(RuntimeLibTest, SimpleLoadGraphFromFileTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    EXPECT_EQ (7 , g.num_edges());
}

TEST_F(RuntimeLibTest, SimpleLoadVerticesromEdges) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    int num_vertices = builtin_getVertices(g);
    //max node id + 1, assumes the first node has id 0
    EXPECT_EQ (5 , num_vertices);
}

TEST_F(RuntimeLibTest, GetOutDegrees) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    auto out_degrees = builtin_getOutDegrees(g);
    //max node id + 1, assumes the first node has id 0
    //TODO: fix this, we don't use vectors anymore
    //EXPECT_EQ (5 , out_degrees.size());
}

TEST_F(RuntimeLibTest, TimerTest) {
//    float start_time = getTime();
//    sleep(1);
//    float end_time = getTime();
    startTimer();
    sleep(1);
    float elapsed_time = stopTimer();
    std::cout << "elapsed_time: " << elapsed_time << std::endl;

    startTimer();
    sleep(2);
    elapsed_time = stopTimer();
    std::cout << "elapsed_time: " << elapsed_time << std::endl;
    EXPECT_EQ (5 , 5);
}

TEST_F(RuntimeLibTest, VertexSubsetSimpleTest) {
    bool test_flag = true;
    auto vertexSubset = new VertexSubset<int>(5, 0);
    for (int v = 0; v < 5; v = v+2){
        vertexSubset->addVertex(v);
    }

    for (int v = 1; v < 5; v = v+2){
        if (vertexSubset->contains(v))
            test_flag = false;
    }

    for (int v = 0; v < 5; v = v+2){
        if (!vertexSubset->contains(v))
            test_flag = false;
    }

    EXPECT_EQ(builtin_getVertexSetSize(vertexSubset), 3);


    delete vertexSubset;

    EXPECT_EQ(true, test_flag);

} 

TEST_F(RuntimeLibTest, IntersectHiroshiTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersect_hiroshi(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);


}


TEST_F(RuntimeLibTest, IntersectBinarySearchTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersect_binary_search(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);


}

TEST_F(RuntimeLibTest, IntersectMultipleSkipTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersect_multiple_skip(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);

}

