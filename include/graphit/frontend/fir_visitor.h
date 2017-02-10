//
// Created by Yunming Zhang on 1/24/17.
//

#ifndef GRAPHIT_FIR_VISITOR_H
#define GRAPHIT_FIR_VISITOR_H


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
            virtual void visit(std::shared_ptr<IntLiteral> op) {} //leaf FIR nodes need no recursive calls
            virtual void visit(std::shared_ptr<AddExpr>);
            virtual void visit(std::shared_ptr<MinusExpr>);


        private:
            //void visitUnaryExpr(std::shared_ptr<UnaryExpr>);
            void visitBinaryExpr(std::shared_ptr<BinaryExpr>);

        };
    }

}
#endif //GRAPHIT_FIR_VISITOR_H