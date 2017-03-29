//
// Created by Yunming Zhang on 2/14/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/midend/midend.h>
#include <graphit/frontend/error.h>

using namespace std;
using namespace graphit;



//tests mid end
TEST(MIRGenTest, SimpleAdd ) {
    Frontend * fe = new Frontend();
    istringstream is("const a = 3 + 4;");
    graphit::FIRContext* fir_context = new graphit::FIRContext();
    std::vector<ParseError> * errors = new std::vector<ParseError>();
    fe->parseStream(is, fir_context, errors);
    graphit::MIRContext* mir_context  = new graphit::MIRContext();
    graphit::Midend* me = new graphit::Midend(fir_context);
    int output = me->emitMIR(mir_context);
    //std::cout << "mir: " << std::endl;
    //std::cout << *(mir_context->getStatements().front());
    //std::cout << std::endl;

    EXPECT_EQ (0 ,  output);
}