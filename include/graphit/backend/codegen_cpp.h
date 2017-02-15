//
// Created by Yunming Zhang on 2/14/17.
//

#ifndef GRAPHIT_CODEGEN_C_H
#define GRAPHIT_CODEGEN_C_H

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>

namespace graphit {
    class CodeGenCPP : mir::MIRVisitor{
    public:

        CodeGenCPP(){

        }

        int genCPP(MIRContext* mir_context);

    protected:
        virtual void visit(mir::Program::Ptr);
        virtual void visit(mir::Stmt::Ptr);
        virtual void visit(mir::Expr::Ptr);
        virtual void visit(mir::AddExpr::Ptr);
        virtual void visit(mir::MinusExpr::Ptr);
        virtual void visit(mir::IntLiteral::Ptr);
    };
}

#endif //GRAPHIT_CODEGEN_C_H
