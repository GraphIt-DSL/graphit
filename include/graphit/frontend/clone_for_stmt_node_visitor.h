//
// Created by Riyadh Baghdadi on 6/26/17.
//

#ifndef GRAPHIT_CLONE_FOR_STMT_NODE_VISITOR_H_H
#define GRAPHIT_CLONE_FOR_STMT_NODE_VISITOR_H_H

#include <graphit/frontend/fir_visitor.h>
#include <graphit/frontend/fir.h>
#include <string>

namespace  graphit {
    namespace  fir {

        struct CloneForStmtNodeVisitor : public fir::FIRVisitor {
            using fir::FIRVisitor::visit;

            CloneForStmtNodeVisitor(){
                target_stmt_ = nullptr;
            }

            // the clone method that returns the loop for stmt
            fir::ForStmt::Ptr cloneForStmtNode(fir::Program::Ptr program, std::string label);

            virtual void visit(fir::ForStmt::Ptr stmt);

        private:
            //this stores the stmt
            fir::ForStmt::Ptr target_stmt_;
            std::string target_label_;
        };
    }
}


#endif //GRAPHIT_CLONE_FOR_STMT_NODE_VISITOR_H_H
