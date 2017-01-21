//
// Created by Yunming Zhang on 1/20/17.
//

#include <gtest.h>
#include <math.h>

double square_root (double input)
{
    return 0;
}

TEST(SquareRootTest, PositiveNos) {
EXPECT_EQ (18.0, square_root (324.0));
EXPECT_EQ (25.4, square_root (645.16));
EXPECT_EQ (50.3321, square_root (2533.310224));
}

TEST (SquareRootTest, ZeroAndNegativeNos) {
ASSERT_EQ (0.0, square_root (0.0));
ASSERT_EQ (-1, square_root (-22.0));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}