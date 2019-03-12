//
// Created by Yunming Zhang on 4/27/17.
//

#include <gtest.h>
#include "intrinsics.h"
#include "infra_gapbs/graph_verifier.h"
#include "infra_gapbs/ordered_processing.h"





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


//test init of the eager priority queue based on GAPBS
TEST_F(RuntimeLibTest, EagerPriorityQueueInit) {

    const WeightT kDistInf = std::numeric_limits<WeightT>::max()/2;

    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/test.el");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        dist_array[i] = kDistInf;
    }
    EagerPriorityQueue<WeightT> pq = EagerPriorityQueue<WeightT>(dist_array);

    EXPECT_EQ(pq.get_current_priority(), 0);
}

//test init of the buffered priority queue based on Julienne
TEST_F(RuntimeLibTest, BufferedPriorityQueueInit) {


    EXPECT_EQ(true, true);
}

// test compilation of the C++ version of SSSP using eager priority queue
TEST_F(RuntimeLibTest, SSSP_test){



    const WeightT kDistInf = std::numeric_limits<WeightT>::max()/2;

    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/test.el");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        dist_array[i] = kDistInf;
    }

    NodeID source = 2;
    dist_array[source] = 0;

    EagerPriorityQueue<WeightT> pq = EagerPriorityQueue<WeightT>(dist_array);
    WeightT input_delta = 2;

    auto src_filter_func = [&](NodeID v)-> bool {
        return (dist_array[v] >= input_delta * static_cast<WeightT>(pq.get_current_priority()));
    };

    auto while_cond_func = [&]()->bool{
            return !pq.finished();
    };


    auto edge_update_func = [&](vector<vector<NodeID> >& local_bins, NodeID src, NodeID dst, WeightT wt) {
        WeightT old_dist = dist_array[dst];
        WeightT new_dist = dist_array[src] + wt;
        update_priority_min<WeightT>()(&pq, local_bins, dst, old_dist, new_dist);
    };

    OrderedProcessingOperatorWithMerge(&pq, g, dist_array, src_filter_func, while_cond_func, edge_update_func, 1000,  source);

    EXPECT_EQ(true, true);

}

// test compilation of the C++ version of PPSP using eager priority queue
TEST_F(RuntimeLibTest, PPSP_test){

}

// test compilation of the C++ version of AStar using eager priority queue
TEST_F(RuntimeLibTest, AStar_test){

}

// test compilation of the C++ version of KCore using buffered priority queue
TEST_F(RuntimeLibTest, KCore_test){

}

