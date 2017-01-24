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
    }
}