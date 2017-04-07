//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    //::testing::GTEST_FLAG(filter) = "FrontendTest.*";
    //::testing::GTEST_FLAG(filter) = "MidendTest.*";
    //::testing::GTEST_FLAG(filter) = "BackendTest.*";

    ::testing::GTEST_FLAG(filter) = "BackendTest.SimpleFunctionWithVarDecl";
    //::testing::GTEST_FLAG(filter) = "MIRGenTest.SimpleFunctionDecl";
//    ::testing::GTEST_FLAG(filter) = "CodeGenTest.SimpleFunctionDecl";



    return RUN_ALL_TESTS();
}