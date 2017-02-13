//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/frontend/mir_emitter.h>

namespace graphit {
    namespace fir {


        void MIREmitter::visit(fir::Program::Ptr program){
//            for (auto elem : program->elems) {
//                elem->accept(this);
//            }
            //ctx->mid_ir = mir_program;
            auto mir_program = std::make_shared<mir::Program>();
            for (auto elem : program->elems) {
                elem->accept(this);
            }

        };

        void MIREmitter::visit(fir::Stmt::Ptr fir_stmt){
            auto mir_stmt = std::make_shared<mir::Stmt>();
            auto expr = this->emitExpr(fir_stmt->expr);
            mir_stmt->expr = expr;
            //useful for later implementing emitStmt
            retStmt = mir_stmt;
            ctx->addStatement(mir_stmt);
        };


        void MIREmitter::visit(AddExpr::Ptr fir_expr){
            auto mir_expr = std::make_shared<mir::AddExpr>();
            mir_expr->lhs = emitExpr(fir_expr->lhs);
            mir_expr->rhs = emitExpr(fir_expr->rhs);
            retExpr = mir_expr;
        };
        void MIREmitter::visit(MinusExpr::Ptr fir_expr){
            auto mir_expr = std::make_shared<mir::MinusExpr>();
            mir_expr->lhs = emitExpr(fir_expr->lhs);
            mir_expr->rhs = emitExpr(fir_expr->rhs);
            retExpr = mir_expr;
        };
        void MIREmitter::visit(IntLiteral::Ptr fir_expr){
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

    }
}