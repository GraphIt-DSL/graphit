//
// Created by Yunming Zhang on 4/27/17.
//

#include <gtest.h>
#include "intrinsics.h"
//#include "infra_gapbs/graph_verifier.h"




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

TEST_F(RuntimeLibTest, serialMSTTest) {
    WGraph wg = builtin_loadWeightedEdgesFromFile("../../test/graphs/mst_special_case.wel");
    NodeID start = 1;
    NodeID* parent_vector = minimum_spanning_tree(wg, start);

    for (int i = 0; i < wg.num_nodes(); i++){
        std::cout << "parent: " << parent_vector[i] << std::endl;
    }

    EXPECT_EQ (1 , parent_vector[1]);
    EXPECT_EQ (3 , parent_vector[4]);
    EXPECT_EQ (4 , parent_vector[5]);
}


TEST_F(RuntimeLibTest, serialMSTTest2) {
    WGraph wg = builtin_loadWeightedEdgesFromFile("../../test/graphs/test2.wel");
    NodeID start = 1;
    NodeID* parent_vector = minimum_spanning_tree(wg, start);

    for (int i = 0; i < wg.num_nodes(); i++){
        std::cout << "parent: " << parent_vector[i] << std::endl;
    }

    EXPECT_EQ (1 , parent_vector[2]);
    EXPECT_EQ (1 , parent_vector[3]);
    EXPECT_EQ (3 , parent_vector[4]);
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

TEST_F(RuntimeLibTest, GetRandomOutNeighborTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    NodeID ngh = g.get_random_out_neigh(1);
    std::cout << ngh << std::endl;
    EXPECT_LE (ngh , 5);
    EXPECT_GE (ngh , 2);
}

TEST_F(RuntimeLibTest, GetRandomInNeighborTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    NodeID ngh = g.get_random_in_neigh(4);
    std::cout << ngh << std::endl;
    EXPECT_LE (ngh , 3);
    EXPECT_GE (ngh , 1);

}


TEST_F(RuntimeLibTest, SweepCutTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    auto vertexSubset = new VertexSubset<int>(g.num_nodes(), g.num_nodes());
    double * val_array = new double[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        val_array[i] = i;
    }
    VertexSubset<int>* vset_cut = serialSweepCut(g, vertexSubset, val_array);

    for (int i = 0; i < vset_cut->size(); i++){
        std::cout << "vertex: " << vset_cut->dense_vertex_set_[i] << std::endl;
    }

    EXPECT_EQ (vset_cut->size() , 2);
}
