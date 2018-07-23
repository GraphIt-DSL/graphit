//
// Created by Yunming Zhang on 2/14/17.
//

#ifndef GRAPHIT_CODEGEN_C_H
#define GRAPHIT_CODEGEN_C_H

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>
#include <iostream>
#include <sstream>
#include <graphit/backend/gen_edge_apply_func_decl.h>

namespace graphit {
    class CodeGenCPP : mir::MIRVisitor{
    public:
        CodeGenCPP(std::ostream &input_oss, MIRContext *mir_context) :
                oss(input_oss), mir_context_(mir_context) {
            indentLevel = 0;
            edgeset_apply_func_gen_ = new EdgesetApplyFunctionDeclGenerator(mir_context_, oss);
        }

        int genCPP();

    protected:

        virtual void visit(mir::ForStmt::Ptr);
        virtual void visit(mir::WhileStmt::Ptr);
        virtual void visit(mir::IfStmt::Ptr);


        virtual void visit(mir::ExprStmt::Ptr);
        virtual void visit(mir::AssignStmt::Ptr);
        virtual void visit(mir::ReduceStmt::Ptr);
        virtual void visit(mir::CompareAndSwapStmt::Ptr);

        virtual void visit(mir::PrintStmt::Ptr);
        virtual void visit(mir::BreakStmt::Ptr);

        virtual void visit(mir::FuncDecl::Ptr);

        virtual void visit(mir::Call::Ptr);

        //virtual void visit(mir::TensorReadExpr::Ptr);
        virtual void visit(mir::TensorStructReadExpr::Ptr);
        virtual void visit(mir::TensorArrayReadExpr::Ptr);

        virtual void visit(mir::VertexSetAllocExpr::Ptr);
        virtual void visit(mir::ListAllocExpr::Ptr);

        //functional operators
        virtual void visit(mir::VertexSetApplyExpr::Ptr);
        virtual void visit(mir::PullEdgeSetApplyExpr::Ptr);
        virtual void visit(mir::PushEdgeSetApplyExpr::Ptr);

        virtual void visit(mir::VertexSetWhereExpr::Ptr);
        //virtual void visit(mir::EdgeSetWhereExpr::Ptr);

        virtual void visit(mir::VarExpr::Ptr);
        virtual void visit(mir::EdgeSetLoadExpr::Ptr);

        virtual void visit(mir::NegExpr::Ptr);
        virtual void visit(mir::EqExpr::Ptr);


        virtual void visit(mir::MulExpr::Ptr);
        virtual void visit(mir::DivExpr::Ptr);
        virtual void visit(mir::AddExpr::Ptr);
        virtual void visit(mir::SubExpr::Ptr);


        virtual void visit(mir::BoolLiteral::Ptr);
        virtual void visit(mir::StringLiteral::Ptr);
        virtual void visit(mir::FloatLiteral::Ptr);
        virtual void visit(mir::IntLiteral::Ptr);


        virtual void visit(mir::VarDecl::Ptr);
        virtual void visit(mir::ElementType::Ptr element_type);

        virtual void visit(mir::VertexSetType::Ptr vertexset_type);
        virtual void visit(mir::ListType::Ptr list_type);

        virtual void visit(mir::StructTypeDecl::Ptr struct_type);
        virtual void visit(mir::ScalarType::Ptr scalar_type);
        virtual void visit(mir::VectorType::Ptr vector_type);

        virtual void visit(mir::EdgeSetType::Ptr edgeset_type);

    private:
        void genIncludeStmts();

        void indent() { ++indentLevel; }
        void dedent() { --indentLevel; }
        void printIndent() { oss << std::string(2 * indentLevel, ' '); }
        void printBeginIndent() { oss << std::string(2 * indentLevel, ' ') << "{" << std::endl; }
        void printEndIndent() { oss << std::string(2 * indentLevel, ' ') << "}"; }
        std::ostream &oss;
        unsigned      indentLevel;

        void genPropertyArrayImplementationWithInitialization(mir::VarDecl::Ptr shared_ptr);

        MIRContext * mir_context_;
        EdgesetApplyFunctionDeclGenerator* edgeset_apply_func_gen_;

        void genElementData();

        void genEdgeSets();

        void genStructTypeDecls();

        // generate the call to the right edgeset apply function with all the arguments
        void genEdgesetApplyFunctionCall(mir::EdgeSetApplyExpr::Ptr apply);

        void genPropertyArrayDecl(mir::VarDecl::Ptr shared_ptr);

        void genPropertyArrayAlloc(mir::VarDecl::Ptr shared_ptr);

        void genScalarDecl(mir::VarDecl::Ptr var_decl);

        void genScalarAlloc(mir::VarDecl::Ptr shared_ptr);
    };
}

#endif //GRAPHIT_CODEGEN_C_H
