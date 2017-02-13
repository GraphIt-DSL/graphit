//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/frontend/mir_emitter.h>

namespace graphit {
    namespace fir {


        void MIREmitter::visit(fir::Program::Ptr program){
            //auto mir_program = std::make_shared<mir::Program>();
            for (auto elem : program->elems) {
                elem->accept(this);
            }
            //ctx->mid_ir = mir_program;
        };

        void MIREmitter::visit(fir::Stmt::Ptr stmt){
            auto mir_stmt = std::make_shared<mir::Stmt>();
            auto expr = this->emitExpr(stmt->expr);
            mir_stmt->expr = expr;
            ctx->addStatement(mir_stmt);
        };



        void MIREmitter::visit(AddExpr::Ptr){

        };
        void MIREmitter::visit(MinusExpr::Ptr){

        };
        void MIREmitter::visit(IntLiteral::Ptr){

        };


        //Currently retExpr seems to be useless, but it might be useful in the future
        mir::Expr::Ptr MIREmitter::emitExpr(FIRNode::Ptr ptr){
            auto tmpExpr = retExpr;
            retExpr = std::make_shared<mir::Expr>();

            ptr->accept(this);
            const mir::Expr::Ptr ret = retExpr;

            retExpr = tmpExpr;
            return ret;
        };

    }
}