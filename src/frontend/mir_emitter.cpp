//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/frontend/mir_emitter.h>
#include <graphit/midend/mir.h>

namespace graphit {
    namespace fir {


        void MIREmitter::visit(Program::Ptr){
            auto program = std::make_shared<mir::Program>();

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