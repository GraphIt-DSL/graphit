//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexsetFilter";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromFilterWithBoolExpression";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleApplyReturnFrontier";


    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexsetFilterComplete";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleFixedIterPageRank";
    //::testing::GTEST_FLAG(filter) = "BackendTest.AddoneWithNoSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.AddoneWithStructSchedule";

    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleApplyReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.VertexSubsetSimpleTest";

    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.TimerTest";

    return RUN_ALL_TESTS();
}