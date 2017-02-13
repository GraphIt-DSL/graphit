//
// Created by Yunming Zhang on 2/9/17.
//

#ifndef GRAPHIT_MIR_EMITTER_H
#define GRAPHIT_MIR_EMITTER_H

#include <graphit/frontend/fir.h>
#include <graphit/frontend/fir_visitor.h>
#include <graphit/midend/program_context.h>

namespace graphit {
    namespace fir {

        class MIREmitter : public FIRVisitor {
        public:
            MIREmitter(internal::ProgramContext* ctx) : ctx(ctx)  {}
            void emitIR(Program::Ptr program) {
                program->accept(this);
            }

            virtual void visit(Program::Ptr);
            virtual void visit(Stmt::Ptr);
            //virtual void visit(Expr::Ptr);
            virtual void visit(AddExpr::Ptr);
            virtual void visit(MinusExpr::Ptr);
            virtual void visit(IntLiteral::Ptr);

            internal::ProgramContext *ctx;

            mir::Expr::Ptr retExpr;
            mir::Stmt::Ptr retStmt;

        private:
            mir::Expr::Ptr     emitExpr(fir::Expr::Ptr);
            mir::Stmt::Ptr     emitStmt(fir::Stmt::Ptr);

        };
    }
}

#endif //GRAPHIT_MIR_EMITTER_H
