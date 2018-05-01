//
// Created by Yunming Zhang on 6/11/17.
//

#ifndef GRAPHIT_CLONELOOPBODYVISITOR_H
#define GRAPHIT_CLONELOOPBODYVISITOR_H

#include <graphit/frontend/fir_visitor.h>
#include <graphit/frontend/fir.h>
#include <string>

namespace graphit {
    namespace  fir {

        struct CloneLoopBodyVisitor : public fir::FIRVisitor {
            using fir::FIRVisitor::visit;

            CloneLoopBodyVisitor(){
                target_loop_body_ = nullptr;
            }

            // the clone method that returns the loop body
            StmtBlock::Ptr CloneLoopBody(fir::Program::Ptr program, std::string label);

            virtual void visit(fir::ForStmt::Ptr stmt);

        private:
            StmtBlock::Ptr target_loop_body_;
            std::string target_label_;
        };

    }
}

#endif //GRAPHIT_CLONELOOPBODYVISITOR_H
