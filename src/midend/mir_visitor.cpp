//
// Created by Yunming Zhang on 2/10/17.
//

//
// Created by Yunming Zhang on 1/24/17.
//


#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir.h>

namespace graphit {
    namespace mir {

        void MIRVisitor::visit(Expr::Ptr expr) {
            expr->accept(this);
        };

        void MIRVisitor::visit(StmtBlock::Ptr stmt_block) {
            for (auto stmt : *(stmt_block->stmts)) {
                stmt->accept(this);
            }
        }

        void MIRVisitor::visit(FuncDecl::Ptr func_decl) {

//            for (auto arg : func_decl->args) {
//                arg->accept(this);
//            }
//            func_decl->result->accept(this);

            if (func_decl->body) {
                func_decl->body->accept(this);
            }
        }

        void MIRVisitor::visit(AddExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRVisitor::visit(SubExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRVisitor::visit(std::shared_ptr<VarDecl> var_decl) {
            var_decl->initVal->accept(this);
        }

        void MIRVisitor::visitBinaryExpr(BinaryExpr::Ptr expr) {
            expr->lhs->accept(this);
            expr->rhs->accept(this);
        }


    }
}