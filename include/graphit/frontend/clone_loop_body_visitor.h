//
// Created by Yunming Zhang on 6/11/17.
//

#ifndef GRAPHIT_CLONELOOPBODYVISITOR_H
#define GRAPHIT_CLONELOOPBODYVISITOR_H

#include <graphit/frontend/fir_visitor.h>
#include <graphit/frontend/fir.h>

namespace graphit {
    namespace  fir {

        struct CloneLoopBodyVisitor : public fir::FIRVisitor {
            using fir::FIRVisitor::visit;

            // the clone method that returns the loop body
            StmtBlock::Ptr CloneLoopBody(fir::Program::Ptr program);

            virtual void visit(fir::ForStmt){
                //if (label_scope_)
            }

        private:
            StmtBlock::Ptr target_loop_body_;
        };

    }
}

#endif //GRAPHIT_CLONELOOPBODYVISITOR_H
