//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/midend/midend.h>


using namespace std;
using namespace graphit;



//tests mid end
TEST(MIRGenTest, SimpleAdd ) {
    Frontend * fe = new Frontend();
    istringstream is("3 + 4;");
    graphit::FIRContext* fir_context = new graphit::FIRContext();
    fe->parseStream(is, fir_context);
    graphit::MIRContext* mir_context  = new graphit::MIRContext();
    graphit::Midend* me = new graphit::Midend(fir_context);
    EXPECT_EQ (0 ,  me->emitMIR(mir_context));
}