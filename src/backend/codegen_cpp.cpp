//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/codegen_cpp.h>

namespace graphit {
    int CodeGenCPP::genCPP(MIRContext *mir_context) {
        //mir_context->getStatements().front()->accept(this);
        for (auto constant : mir_context->getConstants()){
            constant->accept(this);
        }

        return 0;
    };

    void CodeGenCPP::visit(mir::VarDecl::Ptr var_decl){
        stream << var_decl->modifier << ' ';
        var_decl->type->accept(this);
        stream << var_decl->name << " = " ;
        var_decl->initVal->accept(this);
        stream << ";" << std::endl;
    }

    void CodeGenCPP::visit(mir::ScalarType::Ptr scalar_type){
        switch (scalar_type->type) {
            case mir::ScalarType::Type::INT:
                stream << "int ";
                break;
            case mir::ScalarType::Type::FLOAT:
                stream << "float ";
                break;
            case mir::ScalarType::Type::BOOL:
                stream << "bool ";
                break;
            case mir::ScalarType::Type::COMPLEX:
                stream << "complex ";
                break;
            case mir::ScalarType::Type::STRING:
                stream << "string ";
                break;
            default:
                //unreachable;
                break;
        }
    }


    void CodeGenCPP::visit(mir::AddExpr::Ptr expr){
        stream << '(';
        expr->lhs->accept(this);
        stream << " + ";
        expr->rhs->accept(this);
        stream << ')';
    };
    void CodeGenCPP::visit(mir::SubExpr::Ptr expr){
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
