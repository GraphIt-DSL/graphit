//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.MainFunctionWithArgv";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimplePlusReduce";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromToFilterWithFromVertexsetExpression";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleIfElifElseStmt";
    //::testing::GTEST_FLAG(filter) = "BackendTest.VertexSetLibraryCalls";


    //::testing::GTEST_FLAG(filter) = "BackendTest*";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleMinReduceReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleMaxReduce";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleMinReduceReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "BackendTest.MinReduceReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFSPushSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleWeightedEdgeSetApply";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleApplyReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.*";

    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.VertexSubsetSimpleTest";

    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleInsertNameNodeBefore";
    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopFusion";
    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";
    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";

    //::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleApplyFunctionFusion";

    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSPushSlidingQueueSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCHybridDenseSchedule";


    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SSSPPullParallelSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCPullSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallel";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSWithPullParallelSchedule";

    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullVertexsetParallel";

    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSPushParallelSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SSSPPushParallelSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleSerialVertexSetApply";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleParallelVertexSetApply";


    return RUN_ALL_TESTS();
}