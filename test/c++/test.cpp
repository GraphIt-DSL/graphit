//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

//    ::testing::GTEST_FLAG(filter) = "FrontendTest.MainFunctionWithArgv";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.SimplePlusReduce";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromToFilterWithFromVertexsetExpression";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.VectorVertexProperty";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.UninitializedVertexProperty";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.VertexsetFiltering";
//
//
//
//    ::testing::GTEST_FLAG(filter) = "BackendTest*";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleDoubleVarDecl";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVectorSum";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexSetListPop";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SrcFilterDstFilterApply";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFSPushSchedule";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.UninitializedVertexProperty";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.VectorVertexProperty";
//
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexsetFilterComplete";
//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.*";
//
//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.VertexSubsetSimpleTest";
//
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleEdgesetApplyPushSchedule";
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopFusion";
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";
//
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleApplyFunctionFusion";
//
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleBFSWithHyrbidDenseForwardSerialSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCHybridDenseSchedule";
//


//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleLabelForVarDecl";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallelTwoSegments";

//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BCDensePullSparsePushCacheOptimizedSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSPushParallelSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCHybridDenseBitvectorFrontierScheduleNewAPI";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PageRankDeltaDoubleHybridDenseParallelFuseFieldsLoadBalance";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSWithPullParallelSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PageRankDeltaPullParallelFuseFields";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CFPullParallelLoadBalance";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CFPullParallelLoadBalanceWithGrainSize";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PageRankDeltaHybridDenseParallelFuseFieldsLoadBalance";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPushParallel";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSHybridDenseSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SSSPPushParallelSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SimpleSerialVertexSetApply";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.BFSPushSerialSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.SSSPwithHybridDenseSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCHybridDenseBitvectorFrontierSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallelTwoSegments";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCHybridDenseTwoSegments";
//      ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRCCPullParallelDifferentSegments";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallelNumaAware";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallelNumaAware";


    return RUN_ALL_TESTS();
}
