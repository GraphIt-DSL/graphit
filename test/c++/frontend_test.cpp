//
// Created by Yunming Zhang on 1/20/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>


using namespace std;
using namespace graphit;

Frontend * fe = new Frontend();

//tests front end
TEST(LexTest, SimpleAdd ) {
    istringstream is("3 + 4;");
    graphit::FIRContext* context = new graphit::FIRContext();

    EXPECT_EQ (0 ,  fe->parseStream(is, context));
}



