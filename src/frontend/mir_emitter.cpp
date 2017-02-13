//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/frontend/mir_emitter.h>

namespace graphit {
    namespace fir {


        void MIREmitter::visit(Program::Ptr program){
            auto mir_program = std::make_shared<mir::Program>();
            for (auto elem : program->elems) {
                elem->accept(this);
            }
            ctx->mid_ir = mir_program;
        };
        void MIREmitter::visit(Stmt::Ptr){

        };
        void MIREmitter::visit(Expr::Ptr){

        };
        void MIREmitter::visit(AddExpr::Ptr){

        };
        void MIREmitter::visit(MinusExpr::Ptr){

        };
        void MIREmitter::visit(IntLiteral::Ptr){

        };
    }
}