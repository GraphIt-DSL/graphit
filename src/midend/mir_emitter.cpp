//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/midend/mir_emitter.h>

namespace graphit {


    void MIREmitter::visit(fir::ConstDecl::Ptr const_decl){
        addVarOrConst(const_decl, true);
    }


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


    void MIREmitter::addVarOrConst(fir::VarDecl::Ptr var_decl, bool is_const){
        if (is_const){
            const auto mir_var = std::make_shared<mir::VarDecl>();

            mir_var->initVal = emitExpr(var_decl->initVal);
            mir_var->name = var_decl->name->ident;
            mir_var->modifier = "const";
            mir_var->type = emitType(var_decl->type);
            ctx->addConstant(mir_var);
        }
    }



}