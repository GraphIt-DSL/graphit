//
// Created by Yunming Zhang on 6/16/17.
//
#include <graphit/frontend/clone_apply_node_visitor.h>
namespace  graphit {
    namespace  fir {

        fir::ExprStmt::Ptr CloneApplyNodeVisitor::cloneApplyNode(fir::Program::Ptr program,
                                                                 std::string label){
            target_label_ = label;
            program->accept(this);
            return target_expr_stmt_;
        }

        void CloneApplyNodeVisitor::visit(fir::ExprStmt::Ptr stmt) {
            //regular label scoping
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }

            // check if the current label matches the desired label
            if (label_scope_.getCurrentScope() == target_label_) {
                // check if the expr stmt contains an apply expression
                if (fir::isa<fir::ApplyExpr>(stmt->expr)){
                    target_expr_stmt_ = stmt->clone<fir::ExprStmt>();
                }
            }

            //label unscoping
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

    }
}