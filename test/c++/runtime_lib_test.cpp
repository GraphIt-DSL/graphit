//
// Created by Yunming Zhang on 4/27/17.
//

#include <gtest.h>
#include "intrinsics.h"




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
    EXPECT_EQ (5 , out_degrees.size());
}