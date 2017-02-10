//
// Created by Yunming Zhang on 2/9/17.
//

#ifndef GRAPHIT_MIR_EMITTER_H
#define GRAPHIT_MIR_EMITTER_H

#include <graphit/frontend/fir.h>
#include <graphit/frontend/fir_visitor.h>

namespace graphit {
    namespace fir {

        class MIREmitter : public FIRVisitor {
        public:
            MIREmitter() {}
            void emitIR(Program::Ptr program) {program->accept(this);}

            virtual void visit(Program::Ptr);
            virtual void visit(Stmt::Ptr);
            virtual void visit(Expr::Ptr);
            virtual void visit(AddExpr::Ptr);
            virtual void visit(MinusExpr::Ptr);
            virtual void visit(IntLiteral::Ptr);


        };
    }
}

#endif //GRAPHIT_MIR_EMITTER_H
