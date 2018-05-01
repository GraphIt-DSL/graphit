//
// Created by Yunming Zhang on 6/16/17.
//

#ifndef GRAPHIT_CLONE_APPLY_NODE_VISITOR_H_H
#define GRAPHIT_CLONE_APPLY_NODE_VISITOR_H_H

#include <graphit/frontend/fir_visitor.h>
#include <graphit/frontend/fir.h>
#include <string>

namespace  graphit {
    namespace  fir {

        struct CloneApplyNodeVisitor : public fir::FIRVisitor {
            using fir::FIRVisitor::visit;

            CloneApplyNodeVisitor(){
                target_expr_stmt_ = nullptr;
            }

            // the clone method that returns the loop body
            fir::ExprStmt::Ptr cloneApplyNode(fir::Program::Ptr program, std::string label);

            virtual void visit(fir::ExprStmt::Ptr stmt);

        private:
            //this stores the expr stmt that contains the apply expression
            fir::ExprStmt::Ptr target_expr_stmt_;
            std::string target_label_;
        };
    }
}


#endif //GRAPHIT_CLONE_APPLY_NODE_VISITOR_H_H
