//
// Created by Yunming Zhang on 1/20/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/frontend/error.h>

using namespace std;
using namespace graphit;

Frontend * fe = new Frontend();

//tests front end
TEST(LexandParseTest, SimpleVarDecl ) {
    istringstream is("const a : int = 3 + 4;");
    graphit::FIRContext* context = new graphit::FIRContext();
    std::vector<ParseError> * errors = new std::vector<ParseError>();
    int output = fe->parseStream(is, context, errors);
    //prints out the FIR
    std::cout << "fir: " << std::endl;
    std::cout << *context->getProgram();
    std::cout << std::endl;
    EXPECT_EQ (0 ,  output);
}


TEST(LexandParseTest, SimpleFunctionDecl ) {
    istringstream is("func add(a : int, b: int) -> c : int  c = a + b; end");
    graphit::FIRContext* context = new graphit::FIRContext();
    std::vector<ParseError> * errors = new std::vector<ParseError>();
    int output = fe->parseStream(is, context, errors);
    //prints out the FIR
    std::cout << "fir: " << std::endl;
    std::cout << *context->getProgram();
    std::cout << std::endl;
    EXPECT_EQ (0 ,  output);
}


