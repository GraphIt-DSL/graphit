//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.MainFunctionWithArgv";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimplePlusReduce";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromToFilterWithFromVertexsetExpression";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.VectorVertexProperty";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.UninitializedVertexProperty";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.VertexsetFiltering";



    //::testing::GTEST_FLAG(filter) = "BackendTest*";
    //::testing::GTEST_FLAG(filter) = "BackendTest.FabsWithVectorRead";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleMinReduceReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "BackendTest.MinReduceReturnFrontier";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFSPushSchedule";
    //::testing::GTEST_FLAG(filter) = "BackendTest.UninitializedVertexProperty";
    //::testing::GTEST_FLAG(filter) = "BackendTest.VectorVertexProperty";

    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexsetFilterComplete";
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


    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleLoopAndKernelFusion";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCPullSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallel";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSWithPullParallelSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PageRankDeltaPullParallelFuseFields";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CFPullParallelLoadBalance";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CFPullParallelLoadBalanceWithGrainSize";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PageRankDeltaHybridDenseParallelFuseFieldsLoadBalance";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPushParallel";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSHybridDenseSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SSSPPushParallelSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleSerialVertexSetApply";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleParallelVertexSetApply";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SSSPwithHybridDenseSchedule";
    //::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCHybridDenseBitvectorFrontierSchedule";


    return RUN_ALL_TESTS();
}