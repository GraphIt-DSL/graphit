//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleBreak";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleWeightedEdgeSetApply";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromToFilterWithFromVertexsetExpression";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleIfElifElseStmt";
    //::testing::GTEST_FLAG(filter) = "BackendTest.VertexSetLibraryCalls";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleIfElifElseStmt";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleSSSP";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBreak";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgesetApplyPushSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFSPushSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.AddoneWithStructSchedule";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleApplyReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.*";

    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.TimerTest";

    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopFusion";


    return RUN_ALL_TESTS();
}