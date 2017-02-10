//
// Created by Yunming Zhang on 2/9/17.
//

#ifndef GRAPHIT_FIR_PRINTER_H
#define GRAPHIT_FIR_PRINTER_H

#include <iostream>

#include "fir.h"
#include "fir_visitor.h"

namespace graphit {
    namespace fir{
        struct FIRPrinter : public FIRVisitor {
            FIRPrinter(std::ostream &oss) : oss(oss), indentLevel(0) {}

            //void printFIR(fir::Program::Ptr program){program->accept(this);};

        protected:
            virtual void visit(Program::Ptr);
            virtual void visit(Stmt::Ptr);
            virtual void visit(Expr::Ptr);
            virtual void visit(AddExpr::Ptr);
            virtual void visit(MinusExpr::Ptr);
            virtual void visit(IntLiteral::Ptr);

            void printBinaryExpr(BinaryExpr::Ptr expr, const std::string op);


            std::ostream &oss;
            unsigned      indentLevel;

        };
    }
}

#endif //GRAPHIT_FIR_PRINTER_H