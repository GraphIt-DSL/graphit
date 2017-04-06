//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/midend/mir_emitter.h>

namespace graphit {

    void MIREmitter::visit(fir::ConstDecl::Ptr const_decl){
        addVarOrConst(const_decl, true);
    };


    void MIREmitter::visit(fir::AddExpr::Ptr fir_expr){
        auto mir_expr = std::make_shared<mir::AddExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::SubExpr::Ptr fir_expr){
        auto mir_expr = std::make_shared<mir::SubExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::ScalarType::Ptr type){
        switch (type->type) {
            case fir::ScalarType::Type::INT: {
                auto output = std::make_shared<mir::ScalarType>();
                output->type = mir::ScalarType::Type::INT;
                retType = output;
                break;
            }
//            case ScalarType::Type::FLOAT:
//                retType = ir::Float;
//                break;
//            case ScalarType::Type::BOOL:
//                retType = ir::Boolean;
//                break;
//            case ScalarType::Type::COMPLEX:
//                retType = ir::Complex;
//                break;
//            case ScalarType::Type::STRING:
//                retType = ir::String;
//                break;
            default:
                unreachable;
                break;
        }
    }



    void MIREmitter::visit(fir::StmtBlock::Ptr stmt_block){

        //initialzie std:;vector<mir::Stmt> stmts;

        for (auto stmt : stmt_block->stmts) {
            stmt->accept(this);

            //stmts.push_back(emitStmt(stmt))
        }

        auto output_stmt_block = std::make_shared<mir::StmtBlock>();
        if (stmt_block->stmts.size() != 0) {
            output_stmt_block->stmts = ctx->getStatements();
            //output_stmt_block->stmts = stmts;
        }


        retStmt = output_stmt_block;
    }

    // use of variables
    void MIREmitter::visit(fir::VarExpr::Ptr var_expr){
        auto output_var_expr = std::make_shared<mir::VarExpr>();
        mir::Var associated_var = ctx->getSymbol(var_expr->ident);
        output_var_expr->var = associated_var;
        retExpr = output_var_expr;
    }

    void MIREmitter::visit(fir::IdentDecl::Ptr ident_decl){
        auto type = emitType(ident_decl->type);
        //TODO: add type info
        retVar = mir::Var(ident_decl->name->ident, type);
        //TODO: retField in the future ??
    }

    void MIREmitter::visit(fir::FuncDecl::Ptr func_decl){
        auto output_func_decl = std::make_shared<mir::FuncDecl>();
        ctx->scope();
        std::vector<mir::Var> arguments;

        //processing the arguments to the function declaration
        for (auto arg : func_decl->args){
            const mir::Var arg_var = emitVar(arg);
            arguments.push_back(arg_var);
            ctx->addSymbol(arg_var);
        }
        output_func_decl->args = arguments;

        //Processing the output of the function declaration
        //we assume there is only one argument for easy C++ code generation
        assert(func_decl->results.size() <= 1);
        if (func_decl->results.size()){
            const mir::Var result = emitVar(func_decl->results.front());
            output_func_decl->result = result;
        }

        //Processing the body of the function declaration
        mir::Stmt::Ptr body;
        if (func_decl->body != nullptr) {
            body = emitStmt(func_decl->body);



            //Aiming for not using the Scope Node
            //body = ir::Scope::make(body);
        }

        //cast the output to StmtBody
        output_func_decl->body = std::dynamic_pointer_cast<mir::StmtBlock>(body);

        ctx->unscope();

        const auto func_name = func_decl->name->ident;
        output_func_decl->name = func_name;
        //add the constructed function decl to functions
        ctx->addFunction(output_func_decl);
    }

    void MIREmitter::visit(fir::IntLiteral::Ptr fir_expr){
        auto mir_expr = std::make_shared<mir::IntLiteral>();
        mir_expr->val = fir_expr->val;
        retExpr = mir_expr;
    };

    mir::Expr::Ptr MIREmitter::emitExpr(fir::Expr::Ptr ptr){
        auto tmpExpr = retExpr;
        retExpr = std::make_shared<mir::Expr>();

        ptr->accept(this);
        const mir::Expr::Ptr ret = retExpr;

        retExpr = tmpExpr;
        return ret;
    };

    mir::Stmt::Ptr MIREmitter::emitStmt(fir::Stmt::Ptr ptr){
        auto tmpStmt = retStmt;
        retStmt = std::make_shared<mir::Stmt>();

        ptr->accept(this);
        const mir::Stmt::Ptr ret = retStmt;

        retStmt = tmpStmt;
        return ret;
    };

    mir::Type::Ptr MIREmitter::emitType(fir::Type::Ptr ptr) {
        auto tmpType = retType;

        ptr->accept(this);
        auto ret = retType;

        retType = tmpType;
        return ret;
    }


    mir::Var MIREmitter::emitVar(fir::IdentDecl::Ptr ptr) {
        auto tmpVar = retVar;

        ptr->accept(this);
        auto ret = retVar;

        retVar = tmpVar;
        return ret;
    }

    void MIREmitter::addVarOrConst(fir::VarDecl::Ptr var_decl, bool is_const){
        if (is_const){
            //TODO: see if there is a cleaner way to do this
            const auto mir_var = std::make_shared<mir::VarDecl>();
            mir_var->initVal = emitExpr(var_decl->initVal);
            mir_var->name = var_decl->name->ident;
            mir_var->modifier = "const";
            mir_var->type = emitType(var_decl->type);
            ctx->addConstant(mir_var);
        }
    }



}