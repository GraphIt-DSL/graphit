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
    Graph g = loadEdgesFromFile("../../test/graphs/test.el");
    EXPECT_EQ (7 , g.num_edges());
}