//
// Created by Yunming Zhang on 6/11/17.
//

#include <graphit/frontend/clone_loop_body_visitor.h>

namespace graphit {
    namespace  fir {

            StmtBlock::Ptr CloneLoopBodyVisitor::CloneLoopBody(fir::Program::Ptr program) {
                program->accept(this);
                return target_loop_body_;
            };
    }
}