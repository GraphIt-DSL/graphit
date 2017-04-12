//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/codegen_cpp.h>

namespace graphit {
    int CodeGenCPP::genCPP(MIRContext *mir_context) {

        genIncludeStmts();

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


        oss << std::endl;
        return 0;
    };

    void CodeGenCPP::genIncludeStmts(){
        oss << "#include <iostream> " << std::endl;
    }

    void CodeGenCPP::visit(mir::ExprStmt::Ptr expr_stmt){
        printIndent();
        expr_stmt->expr->accept(this);
        oss << ";" << std::endl;
    }

    void CodeGenCPP::visit(mir::AssignStmt::Ptr assign_stmt){
        printIndent();
        assign_stmt->lhs->accept(this);
        oss << " = ";
        assign_stmt->expr->accept(this);
        oss << ";" << std::endl;
    }

    void CodeGenCPP::visit(mir::PrintStmt::Ptr print_stmt){
        printIndent();
        oss << "std::cout << ";
        print_stmt->expr->accept(this);
        oss << "<< std::endl;" << std::endl;
    }

    void CodeGenCPP::visit(mir::VarDecl::Ptr var_decl){
        printIndent();
        oss << var_decl->modifier << ' ';
        var_decl->type->accept(this);
        oss << var_decl->name << " " ;
        if (var_decl->initVal != nullptr){
            oss << "= ";
            var_decl->initVal->accept(this);
        }
        oss << ";" << std::endl;
    }



    void CodeGenCPP::visit(mir::FuncDecl::Ptr func_decl){

        //generate the return type
        if (func_decl->result.isInitialized()) {
            func_decl->result.getType()->accept(this);

            //insert an additional var_decl for returning result
            const auto var_decl = std::make_shared<mir::VarDecl>();
            var_decl->name = func_decl->result.getName();
            var_decl->type = func_decl->result.getType();
            if (func_decl->body->stmts == nullptr){
                func_decl->body->stmts = new std::vector<mir::Stmt::Ptr>();
            }
            auto it = func_decl->body->stmts->begin();
            func_decl->body->stmts->insert(it, var_decl);

        } else if (func_decl->name == "main"){
            oss << "int ";
        } else {
            //default to int return type
            oss << "void ";
        }

        //generate the function name and left paren
        oss << func_decl->name << "(";

        bool printDelimiter = false;
        for (auto arg : func_decl->args) {
            if (printDelimiter) {
                oss << ", ";
            }

            arg.getType()->accept(this);
            oss << arg.getName();
            printDelimiter = true;
        }
        oss << ") ";



        //if the function has a body
        if (func_decl->body->stmts) {
            oss << std::endl;
            printBeginIndent();
            indent();

            func_decl->body->accept(this);

            //print a return statemetn if there is a result
            if(func_decl->result.isInitialized()){
                printIndent();
                oss << "return " << func_decl->result.getName() << ";" << std::endl;
            }

            dedent();
            printEndIndent();
        }
        oss << ";";
        oss << std::endl;

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
            case mir::ScalarType::Type::STRING:
                oss << "string ";
                break;
            default:
                break;
        }
    }

    void CodeGenCPP::visit(mir::Call::Ptr call_expr){
        oss << call_expr->name << "(";

        bool printDelimiter = false;

        for (auto arg : call_expr->args){
            if (printDelimiter) {
                oss << ", ";
            }
            arg->accept(this);
            printDelimiter = true;
        }

        oss << ") ";
    };

    void CodeGenCPP::visit(mir::VarExpr::Ptr expr){
        oss << expr->var.getName();
    };

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
        //oss << "(int) ";
        oss << expr->val;
        oss << ") ";
    };



}
