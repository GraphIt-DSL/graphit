//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.*";
    //::testing::GTEST_FLAG(filter) = "MidendTest.*";
    //::testing::GTEST_FLAG(filter) = "BackendTest.*";

    //::testing::GTEST_FLAG(filter) = "FrontendTest.MainFunctionWithPrint";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.MainFunctionWithCall";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVecDecl";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexSet";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexSetAlloc";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexSetDeclAlloc";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleEdgeSetWithMain";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVariable";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVectorSum";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleMethodCallsChaining";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleTensorRead";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexSetApply";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleVertexSetLoad";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleEdgeSetApply";
    //::testing::GTEST_FLAG(filter) = "FrontendTest.SimpleForLoops";


    //::testing::GTEST_FLAG(filter) = "MidendTest.MainFunctionWithPrint";
    //::testing::GTEST_FLAG(filter) = "MidendTest.MainFunctionWithCall";
    //::testing::GTEST_FLAG(filter) = "MidendTest.SimpleVertexSetDeclAlloc";


    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleFunctionWithVarDecl";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleFunctionWithAdd";
    //::testing::GTEST_FLAG(filter) = "BackendTest.MainFunctionWithPrint";
    //::testing::GTEST_FLAG(filter) = "BackendTest.MainFunctionWithCall";
    //::testing::GTEST_FLAG(filter) = "BackendTest.ElementDecl";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexSetDeclAlloc";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgeSetWithMain";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVariable";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVectorSum";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexSetApply";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgeSetLoad";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleEdgeSetApply";
    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleForLoops";


    //::testing::GTEST_FLAG(filter) = "BackendTest.SimpleVertexSetLoad";

    //::testing::GTEST_FLAG(filter) = "RuntimeLibTest.SimpleLoadGraphFromFileTest";


    return RUN_ALL_TESTS();
}