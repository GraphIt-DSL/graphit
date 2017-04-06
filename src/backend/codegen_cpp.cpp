//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/codegen_cpp.h>

namespace graphit {
    int CodeGenCPP::genCPP(MIRContext *mir_context) {

        //Processing the constants
        for (auto constant : mir_context->getConstants()){
            constant->accept(this);
        }

        //Processing the functions
        std::map<std::string, mir::FuncDecl::Ptr>::iterator it;
        auto functions = mir_context->getFunctions();

        for ( it = functions.begin(); it != functions.end(); it++ )
        {
            it->second->accept(this);
        }

        return 0;
    };

    void CodeGenCPP::visit(mir::VarDecl::Ptr var_decl){
        oss << var_decl->modifier << ' ';
        var_decl->type->accept(this);
        oss << var_decl->name << " = " ;
        var_decl->initVal->accept(this);
        oss << ";" << std::endl;
    }



    void CodeGenCPP::visit(mir::FuncDecl::Ptr func_decl){



    };

    void CodeGenCPP::visit(mir::ScalarType::Ptr scalar_type){
        switch (scalar_type->type) {
            case mir::ScalarType::Type::INT:
                oss << "int ";
                break;
            case mir::ScalarType::Type::FLOAT:
                oss << "float ";
                break;
            case mir::ScalarType::Type::BOOL:
                oss << "bool ";
                break;
            case mir::ScalarType::Type::COMPLEX:
                oss << "complex ";
                break;
            case mir::ScalarType::Type::STRING:
                oss << "string ";
                break;
            default:
                //unreachable;
                break;
        }
    }


    void CodeGenCPP::visit(mir::AddExpr::Ptr expr){
        oss << '(';
        expr->lhs->accept(this);
        oss << " + ";
        expr->rhs->accept(this);
        oss << ')';
    };
    void CodeGenCPP::visit(mir::SubExpr::Ptr expr){
        oss << '(';
        expr->lhs->accept(this);
        oss << " - ";
        expr->rhs->accept(this);
        oss << ')';
    };
    void CodeGenCPP::visit(mir::IntLiteral::Ptr expr){
        oss << "(";
        oss << "(int) ";
        oss << expr->val;
        oss << ")";
    };



}
