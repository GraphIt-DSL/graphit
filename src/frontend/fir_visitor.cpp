//
// Created by Yunming Zhang on 1/24/17.
//
#include <graphit/frontend/fir.h>

namespace graphit {
    namespace fir {

        void FIRVisitor::visit(Program::Ptr program) {
            for (auto elem : program->elems) {
                elem->accept(this);
            }
        }

        void FIRVisitor::visit(Stmt::Ptr stmt) {
            stmt->accept(this);
        };

        void FIRVisitor::visit(Expr::Ptr expr) {
            expr->accept(this);
        };

    }
}