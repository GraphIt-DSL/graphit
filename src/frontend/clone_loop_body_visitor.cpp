//
// Created by Yunming Zhang on 6/11/17.
//

#include <graphit/frontend/clone_loop_body_visitor.h>

namespace graphit {
    namespace fir {

        StmtBlock::Ptr CloneLoopBodyVisitor::CloneLoopBody(fir::Program::Ptr program,
                                                           std::string label) {
            target_label_ = label;
            program->accept(this);
            return target_loop_body_;
        }

        void CloneLoopBodyVisitor::visit(fir::ForStmt::Ptr stmt) {
            //regular label scoping
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }

            // check if the current label matches the desired label
            if (label_scope_.getCurrentScope() == target_label_) {
                target_loop_body_ = fir::to<StmtBlock>(stmt->body->clone());
            }

            //label unscoping
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }
    }
}