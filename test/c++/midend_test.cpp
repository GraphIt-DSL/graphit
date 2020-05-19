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


class MidendTest : public ::testing::Test {
protected:
    virtual void SetUp(){
        context_ = new graphit::FIRContext();
        errors_ = new std::vector<ParseError>();
        fe_ = new Frontend();
        mir_context_  = new graphit::MIRContext();

    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).

        //prints out the MIR, just a hack for now
        //std::cout << "mir: " << std::endl;
        //std::cout << *(mir_context->getStatements().front());
        //std::cout << std::endl;

    }

    bool basicTest(std::istream & is){
        fe_->parseStream(is, context_, errors_);
        graphit::Midend* me = new graphit::Midend(context_);
        return me->emitMIR(mir_context_);
    }


    std::vector<ParseError> * errors_;
    graphit::FIRContext* context_;
    Frontend * fe_;
    graphit::MIRContext* mir_context_;
};


//tests mid end
TEST_F(MidendTest, SimpleVarDecl ) {
    istringstream is("const a : int = 3 + 4;");
    EXPECT_EQ (0 , basicTest(is));
}


//tests mid end
TEST_F(MidendTest, SimpleFunctionDecl) {
    istringstream is("func add(a : int, b: int) -> c : int  end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, SimpleFunctionDeclWithNoReturn) {
    istringstream is("func add(a : int, b: int)  end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, SimpleFunctionWithVarDecl) {
    istringstream is("func add(a : int, b: int) -> c : int var d : int = 3; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, SimpleFunctionWithAdd) {
    istringstream is("func add(a : int, b: int) -> c : int c = a + b; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, MainFunctionWithPrint) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ (0 , basicTest(is));
}

TEST_F(MidendTest, MainFunctionWithCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, MainFunctionWithPrintCall) {
    istringstream is("func add(a : int, b: int) -> c:int c = a + b; end\n"
                             " func main() print add(4, 5); end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, ElementDecl) {
    istringstream is("element Vertex end");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, SimpleVertexSetDeclAlloc) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);");
    EXPECT_EQ (0,  basicTest(is));
}

TEST_F(MidendTest, SimpleVertexSetDeclAllocWithMain) {
    istringstream is("element Vertex end\n"
                             "const vector_a : vector{Vertex}(float) = 0.0;\n"
                             "const vertices : vertexset{Vertex} = new vertexset{Vertex}(5);\n"
                             "func main() print 4; end");
    EXPECT_EQ (0, basicTest(is));
}

// Test cases for the MIRMetadata API
TEST_F(MidendTest, SimpleMetadataTest) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ(0, basicTest(is));
    EXPECT_EQ(true, mir_context_->isFunction("main"));

    mir::FuncDecl::Ptr main_func = mir_context_->getFunction("main");

    main_func->setMetadata<bool>("basic_boolean_md", true);
    main_func->setMetadata<int>("basic_int_md", 42);
    EXPECT_EQ(true, main_func->hasMetadata<bool>("basic_boolean_md"));
    EXPECT_EQ(true, main_func->getMetadata<bool>("basic_boolean_md"));
    
    EXPECT_EQ(true, main_func->hasMetadata<int>("basic_int_md"));
    EXPECT_EQ(42, main_func->getMetadata<int>("basic_int_md"));

}
TEST_F(MidendTest, SimpleMetadataTestNoExist) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ(0, basicTest(is));
    EXPECT_EQ(true, mir_context_->isFunction("main"));

    mir::FuncDecl::Ptr main_func = mir_context_->getFunction("main");

    main_func->setMetadata<int>("basic_int_md", 42);
    EXPECT_EQ(false, main_func->hasMetadata<int>("other_int_md"));
    EXPECT_EQ(false, main_func->hasMetadata<bool>("basic_int_md")); 
}

TEST_F(MidendTest, SimpleMetadataTestString) {
    istringstream is("func main() print 4; end");
    EXPECT_EQ(0, basicTest(is));
    EXPECT_EQ(true, mir_context_->isFunction("main"));

    mir::FuncDecl::Ptr main_func = mir_context_->getFunction("main");

    main_func->setMetadata<std::string>("basic_str_md", "md value");
    EXPECT_EQ(true, main_func->hasMetadata<std::string>("basic_str_md"));
    EXPECT_EQ("md value", main_func->getMetadata<std::string>("basic_str_md"));
}

TEST_F(MidendTest, SimpleMetadataTestMIRNodeAsMD) {
    istringstream is("const val:int = 42;\nfunc main() print val; end");
    EXPECT_EQ(0, basicTest(is));
    EXPECT_EQ(true, mir_context_->isFunction("main"));
    EXPECT_EQ(1, mir_context_->getConstants().size());
     
    mir::FuncDecl::Ptr main_func = mir_context_->getFunction("main");
    mir::VarDecl::Ptr decl = mir_context_->getConstants()[0];

    main_func->setMetadata<mir::MIRNode::Ptr>("used_var_md", decl);
    
    EXPECT_EQ(true, main_func->hasMetadata<mir::MIRNode::Ptr>("used_var_md"));
    mir::MIRNode::Ptr mdnode = main_func->getMetadata<mir::MIRNode::Ptr>("used_var_md");
    EXPECT_EQ(true, mir::isa<mir::VarDecl>(mdnode)); 
}

TEST_F(MidendTest, SimpleMetadataTestMIRNodeVectorAsMD) {
    istringstream is("const val:int = 42;\nconst val2: int = 55;\nfunc main() print val + val2; end");
    EXPECT_EQ(0, basicTest(is));
    EXPECT_EQ(true, mir_context_->isFunction("main"));
    EXPECT_EQ(2, mir_context_->getConstants().size());
     
    mir::FuncDecl::Ptr main_func = mir_context_->getFunction("main");
    std::vector<mir::VarDecl::Ptr> decls = mir_context_->getConstants();

    main_func->setMetadata<std::vector<mir::VarDecl::Ptr>>("used_vars_md", decls);
    
    EXPECT_EQ(true, main_func->hasMetadata<std::vector<mir::VarDecl::Ptr>>("used_vars_md"));
    EXPECT_EQ(2, main_func->getMetadata<std::vector<mir::VarDecl::Ptr>>("used_vars_md").size());
}
