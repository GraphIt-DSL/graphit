//
// Created by Yunming Zhang on 1/24/17.
//

#ifndef GRAPHIT_FIR_VISITOR_H
#define GRAPHIT_FIR_VISITOR_H

#endif //GRAPHIT_FIR_VISITOR_H
#include <memory>

namespace graphit {
    namespace fir {

        struct Program;
        struct Stmt;
        struct Expr;
        struct IntLiteral;
        struct AddExpr;
        struct MinusExpr;
        struct BinaryExpr;

        struct FIRVisitor {
            virtual void visit(std::shared_ptr<Program>);
            virtual void visit(std::shared_ptr<Stmt>);
            virtual void visit(std::shared_ptr<Expr>);
            virtual void visit(std::shared_ptr<IntLiteral> op) {}
            virtual void visit(std::shared_ptr<AddExpr> op);
            virtual void visit(std::shared_ptr<MinusExpr> op);
        private:
            //void visitUnaryExpr(std::shared_ptr<UnaryExpr>);
            void visitBinaryExpr(std::shared_ptr<BinaryExpr>);

        };
    }

}