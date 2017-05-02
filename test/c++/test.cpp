//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.*";
    //::testing::GTEST_FLAG(filter) = "MidendTest.*";
    //::testing::GTEST_FLAG(filter) = "BackendTest.*";

    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleTensorRead";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexSetApply";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexSetLoad";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleEdgeSetApply";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleForLoops";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.VertexSetGetSize";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.EdgeSetGetOutDegrees";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleFixedIterPageRank";


    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgeSetLoad";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgeSetApply";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleForLoops";
    //::testing::GTEST_FLAG(filter) = "BackendTest.VertexSetGetSize";
    //::testing::GTEST_FLAG(filter) = "BackendTest.EdgeSetGetOutDegrees";
    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleFixedIterPageRank";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexSetLoad";

    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.GetOutDegrees";


    return RUN_ALL_TESTS();
}