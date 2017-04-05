//
// Created by Yunming Zhang on 2/13/17.
//

#ifndef GRAPHIT_MIR_PRINTER_H
#define GRAPHIT_MIR_PRINTER_H
#include <iostream>
#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>

namespace graphit {
    namespace mir {
        struct MIRPrinter : public MIRVisitor {
            MIRPrinter(std::ostream &oss) : oss(oss), indentLevel(0) {}

            //void printMIR(MIR::Program::Ptr program){program->accept(this);};

        protected:


            //Delayed for later
//            virtual void visit(VarDecl::Ptr);
//            virtual void visit(FuncDecl::Ptr);


            virtual void visit(Expr::Ptr);
            virtual void visit(AddExpr::Ptr);
            virtual void visit(SubExpr::Ptr);
            virtual void visit(IntLiteral::Ptr);


            void indent() { ++indentLevel; }
            void dedent() { --indentLevel; }
            void printIndent() { oss << std::string(2 * indentLevel, ' '); }

            void printBinaryExpr(BinaryExpr::Ptr expr, const std::string op);


            std::ostream &oss;
            unsigned      indentLevel;

        };
    }
}

#endif //GRAPHIT_MIR_PRINTER_H
