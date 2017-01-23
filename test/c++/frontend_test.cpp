//
// Created by Yunming Zhang on 1/20/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>


using namespace std;
using namespace graphit;

Frontend * fe = new Frontend();

TEST(LexTest, SimpleAdd ) {
    istringstream is("int a = b + c");
    EXPECT_EQ (0 ,  fe->parseStream(is));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}