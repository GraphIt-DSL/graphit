//
// Created by Riyadh Baghdadi on 6/26/17.
//
#include <graphit/frontend/clone_for_stmt_node_visitor.h>
namespace  graphit {
    namespace  fir {

        fir::ForStmt::Ptr CloneForStmtNodeVisitor::cloneForStmtNode(fir::Program::Ptr program,
                                                                    std::string label){
            target_label_ = label;
            program->accept(this);
            return target_stmt_;
        }

        void CloneForStmtNodeVisitor::visit(fir::ForStmt::Ptr stmt) {
            //regular label scoping
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }

            // check if the current label matches the desired label
            if (label_scope_.getCurrentScope() == target_label_) {
                // check if the stmt contains a domain
                if (fir::isa<fir::ForDomain>(stmt->domain)){
                    target_stmt_ = fir::to<fir::ForStmt>(stmt->clone());
                }
            }

            //label unscoping
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

    }
}
