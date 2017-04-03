//
// Created by Yunming Zhang on 2/10/17.
//

#ifndef GRAPHIT_MIR_VISITOR_H
#define GRAPHIT_MIR_VISITOR_H

#include <memory>

namespace graphit {
    namespace mir {

        struct Program;
        struct Stmt;
        struct StmtBlock;
        struct Expr;
        struct IntLiteral;
        struct AddExpr;
        struct SubExpr;
        struct BinaryExpr;
        struct Type;
        struct ScalarType;
        struct VarDecl;
        struct Identifier;
        struct IdentDecl;
        struct FuncDecl;

        struct MIRVisitor {
            virtual void visit(std::shared_ptr<Stmt>){};
            virtual void visit(std::shared_ptr<StmtBlock>);
            virtual void visit(std::shared_ptr<Expr>);
            virtual void visit(std::shared_ptr<IntLiteral> op) {} //leaf FIR nodes need no recursive calls
            virtual void visit(std::shared_ptr<AddExpr>);
            virtual void visit(std::shared_ptr<SubExpr>);
            virtual void visit(std::shared_ptr<Type>){};
            virtual void visit(std::shared_ptr<ScalarType>){};
            virtual void visit(std::shared_ptr<VarDecl>);
            virtual void visit(std::shared_ptr<Identifier>){};
            virtual void visit(std::shared_ptr<IdentDecl>){};
            virtual void visit(std::shared_ptr<FuncDecl>);


        private:
            void visitBinaryExpr(std::shared_ptr<BinaryExpr>);
        };
    }
}


#endif //GRAPHIT_MIR_VISITOR_H
