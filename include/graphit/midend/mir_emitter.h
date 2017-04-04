// Created by Yunming Zhang on 2/9/17.
//

#ifndef GRAPHIT_MIR_EMITTER_H
#define GRAPHIT_MIR_EMITTER_H

#include <graphit/frontend/fir.h>
#include <graphit/frontend/fir_visitor.h>
#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>

namespace graphit {


        class MIREmitter : public fir::FIRVisitor {
        public:
            MIREmitter(MIRContext* ctx) : ctx(ctx)  {}
            ~MIREmitter()  {}

            void emitIR(fir::Program::Ptr program) {
                program->accept(this);
            }

            virtual void visit(fir::ConstDecl::Ptr);

            virtual void visit(fir::StmtBlock);
            virtual void visit(fir::Identifier);
            virtual void visit(fir::IdentDecl);

            virtual void visit(fir::FuncDecl::Ptr);


            virtual void visit(fir::AddExpr::Ptr);
            virtual void visit(fir::SubExpr::Ptr);
            virtual void visit(fir::IntLiteral::Ptr);


            virtual void visit(fir::ScalarType::Ptr);


            MIRContext *ctx;

            mir::Expr::Ptr retExpr;
            mir::Stmt::Ptr retStmt;
            mir::Type::Ptr retType;

        private:
            //helper methods for the visitor pattern to return MIR nodes
            mir::Expr::Ptr     emitExpr(fir::Expr::Ptr);
            mir::Stmt::Ptr     emitStmt(fir::Stmt::Ptr);
            mir::Type::Ptr     emitType(fir::Type::Ptr);

            void addVarOrConst(fir::VarDecl::Ptr var_decl, bool is_const);

        };

}

#endif //GRAPHIT_MIR_EMITTER_H

