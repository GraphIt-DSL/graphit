//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/codegen_cpp.h>

namespace graphit {
    int CodeGenCPP::genCPP(MIRContext *mir_context) {
        mir_context->getStatements().front()->accept(this);
        return 0;
    };


    void CodeGenCPP::visit(mir::Stmt::Ptr stmt){
        stmt->expr->accept(this);
        stream << ";" << std::endl;
    };

    void CodeGenCPP::visit(mir::AddExpr::Ptr expr){
        stream << '(';
        expr->lhs->accept(this);
        stream << " + ";
        expr->rhs->accept(this);
        stream << ')';
    };
    void CodeGenCPP::visit(mir::MinusExpr::Ptr expr){
        stream << '(';
        expr->lhs->accept(this);
        stream << " - ";
        expr->rhs->accept(this);
        stream << ')';
    };
    void CodeGenCPP::visit(mir::IntLiteral::Ptr expr){
        stream << "(";
        stream << "(int) ";
        stream << expr->val;
        stream << ")";
    };



}
