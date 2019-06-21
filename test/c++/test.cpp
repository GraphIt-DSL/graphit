//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);


//    ::testing::GTEST_FLAG(filter) = "FrontendTest.DeltaStepping";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.PointToPointShortestPath";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.MainFunctionWithPrint";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.EdgeSetExportFuncVectorInit";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleApplyFromToFilterWithFromVertexsetExpression";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.VectorVertexProperty";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.UninitializedVertexProperty";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.PriorityQueueApplyUpdatePriority";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.PriorityQueueAllocation";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.GlobalPriorityQueueAllocation";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.KCoreFrontendTest";
//    ::testing::GTEST_FLAG(filter) = "FrontendTest.SetCoverFrontendTest";
//
//
//
//    ::testing::GTEST_FLAG(filter) = "BackendTest*";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.edgesetApplyExtern";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexSetListPop";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SrcFilterDstFilterApply";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleBFSPushSchedule";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.UninitializedVertexProperty";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.VectorVertexProperty";
//
//    ::testing::GTEST_FLAG(filter) = "BackendTest.GlobalConstantSizeVectorTest";
//    ::testing::GTEST_FLAG(filter) = "BackendTest.vectorPerVertexTestWithConstDef";
//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";
//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.*";
//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.AStar_load_graph";

//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.*";
//
//    ::testing::GTEST_FLAG(filter) = "RuntimeLibTest.UpdateAndGetGraphItVertexSubsetFromJulienneBucketsWithUpdatesTest";

//
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleEdgesetApplyPushSchedule";
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopFusion";
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleLoopIndexSplit";
//
//    ::testing::GTEST_FLAG(filter) = "LowLevelScheduleTest.SimpleApplyFunctionFusion";
//
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.AStarDeltaSteppingWithEagerPriorityUpdateWithMergeArgv";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.CCHybridDenseSchedule";
//
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.KCoreSumReduceBeforeUpdate";



//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.KCoreDefaultSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.KCoreSumReduceBeforeUpdate";
    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.DeltaSteppingWithDefaultSchedule";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.DeltaSteppingWithEagerPriorityUpdate";

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
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRCCPullParallelDifferentSegments";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallelNumaAware";
//    ::testing::GTEST_FLAG(filter) = "HighLevelScheduleTest.PRPullParallelNumaAware";


    return RUN_ALL_TESTS();
}
