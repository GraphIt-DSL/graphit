//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyWithModifiedTracking";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleWeightedEdgeSetApply";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromToFilterWithFromVertexsetExpression";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleIfElifElseStmt";
    //::testing::GTEST_FLAG(filter) = "BackendTest.VertexSetLibraryCalls";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleModifiedReturnFrontier";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFS";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBreak";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgesetApplyPushSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFSPushSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.AddoneWithStructSchedule";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleApplyReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.*";

    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.TimerTest";

    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleFunctionFusion";
    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopFusion";
    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";
    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";

    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleApplyFunctionFusion";

    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleStructHighLevelSchedule";

    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.EdgeSetGetOutDegreesFuseStruct";


    return RUN_ALL_TESTS();
}