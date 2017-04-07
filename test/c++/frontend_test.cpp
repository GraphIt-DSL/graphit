//
// Created by Yunming Zhang on 1/20/17.
//

#include <gtest.h>
#include <graphit/frontend/frontend.h>
#include <graphit/midend/mir_context.h>
#include <graphit/frontend/error.h>

using namespace std;
using namespace graphit;


class FrontendTest : public ::testing::Test {
protected:
    virtual void SetUp(){
        context_ = new graphit::FIRContext();
        errors_ = new std::vector<ParseError>();
        fe_ = new Frontend();
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

//        //prints out the FIR, just a hack for now
//        std::cout << "fir: " << std::endl;
//        std::cout << *context_->getProgram();
//        std::cout << std::endl;
    }

    std::vector<ParseError> * errors_;
    graphit::FIRContext* context_;
    Frontend * fe_;
};




//tests front end
TEST_F(FrontendTest, SimpleVarDecl ) {
    istringstream is("const a : int = 3 + 4;");
    int output = fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0 ,  output);
}


TEST_F(FrontendTest, SimpleFunctionDecl ) {
    istringstream is("func add(a : int, b: int) -> c : int  end");
    int output = fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0 ,  output);
}


TEST_F(FrontendTest, SimpleFunctionDeclWithNoReturn ) {
    istringstream is("func add(a : int, b: int) end");
    int output = fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0 ,  output);
}

TEST_F(FrontendTest, SimpleFunctionDecFail) {
    istringstream is("func add(a : int, b: int) ");
    int output = fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (1 ,  output);
}

TEST_F(FrontendTest, SimpleFunctionDecFailNoEnd) {
    istringstream is("func add(a : int, b: int) -> c : int ");
    int output = fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (1 ,  output);
}

TEST_F(FrontendTest, SimpleFunctionWithVarDecl) {
    istringstream is("func add(a : int, b: int) -> c : int var d : int = 3; end");
    int output = fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0 ,  output);
}

TEST_F(FrontendTest, SimpleFunctionWithAdd) {
    istringstream is("func add(a : int, b: int) -> c : int c = a + b; end");
    int output = fe_->parseStream(is, context_, errors_);
    EXPECT_EQ (0, output);
}