//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleWhileLoop";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromToFilterWithFromVertexsetExpression";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleBFS";
    //::testing::GTEST_FLAG(filter) = "BackendTest.VertexSetLibraryCalls";


    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgeSetApply";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleForEdgesetApplyNoNestedLabelsPullSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgesetApplyPushSchedule";
    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFSPushSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.AddoneWithStructSchedule";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleApplyReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.VertexSubsetSimpleTest";

    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.TimerTest";

    return RUN_ALL_TESTS();
}