//
// Created by Yunming Zhang on 4/27/17.
//

#include <gtest.h>
#include "intrinsics.h"
#include "infra_gapbs/graph_verifier.h"
#include "infra_gapbs/graph_relabel.h"
#include "relabel_verifier.h"


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


TEST_F(RuntimeLibTest, RelabelByIndegree) {
    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/test.wel"); 
    pvector<NodeID> new_ids(g.num_nodes());
    auto indegrees_original = builtin_getInDegrees(g);
    std::sort(indegrees_original, indegrees_original + g.num_nodes(), greater<int>());
    auto relabeled_g = relabelByIndegree(g, new_ids);
    auto indegrees_relabeled = builtin_getInDegrees(relabeled_g);
    
    EXPECT_EQ(indegrees_original[0], indegrees_relabeled[0]);
    EXPECT_EQ(indegrees_original[1], indegrees_relabeled[1]);
    EXPECT_EQ(indegrees_original[2], indegrees_relabeled[2]);
    EXPECT_EQ(indegrees_original[3], indegrees_relabeled[3]);
    EXPECT_EQ(new_ids[1], 3);
    EXPECT_EQ(new_ids[4], 0);
    EXPECT_EQ(new_ids[0], 4);

}

TEST_F(RuntimeLibTest, SSSPWithRelabel) {
    WGraph original_g = builtin_loadWeightedEdgesFromFile("../../test/graphs/4.wel"); 
    pvector<NodeID> new_ids(original_g.num_nodes());
    auto relabeled_g = relabelByIndegree(original_g, new_ids);
    auto source = 13;
    auto relabeled_source = new_ids[source];
    int64_t sum_original = GetShortestDistSum(original_g, source);
    int64_t sum_relabeled = GetShortestDistSum(relabeled_g, relabeled_source);
    bool test_flag = sum_original == sum_relabeled;
    EXPECT_EQ(test_flag, true);

}

