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
        struct ExprStmt;
        struct PrintStmt;
        struct AssignStmt;
        struct StmtBlock;
        struct Expr;
        struct FloatLiteral;
        struct IntLiteral;
        struct Call;
        struct VarExpr;
        struct AddExpr;
        struct SubExpr;
        struct BinaryExpr;
        struct Type;
        struct ScalarType;
        struct VarDecl;
        struct IdentDecl;
        struct FuncDecl;
        struct ElementType;
        struct VertexSetType;
        struct VectorType;

        struct MIRVisitor {
            virtual void visit(std::shared_ptr<Stmt>){};
            virtual void visit(std::shared_ptr<AssignStmt>);
            virtual void visit(std::shared_ptr<PrintStmt>);
            virtual void visit(std::shared_ptr<ExprStmt>);
            virtual void visit(std::shared_ptr<StmtBlock>);
            virtual void visit(std::shared_ptr<Expr>);
            virtual void visit(std::shared_ptr<Call>);
            virtual void visit(std::shared_ptr<FloatLiteral>){}
            virtual void visit(std::shared_ptr<IntLiteral> op) {} //leaf FIR nodes need no recursive calls
            virtual void visit(std::shared_ptr<VarExpr>){};
            virtual void visit(std::shared_ptr<AddExpr>);
            virtual void visit(std::shared_ptr<SubExpr>);
            virtual void visit(std::shared_ptr<Type>){};
            virtual void visit(std::shared_ptr<ScalarType>){};
            virtual void visit(std::shared_ptr<VarDecl>);
            virtual void visit(std::shared_ptr<IdentDecl>){};
            virtual void visit(std::shared_ptr<FuncDecl>);
            virtual void visit(std::shared_ptr<ElementType>){};
            virtual void visit(std::shared_ptr<VertexSetType>){};
            virtual void visit(std::shared_ptr<VectorType>){};


        private:
            void visitBinaryExpr(std::shared_ptr<BinaryExpr>);
        };
    }
}


#endif //GRAPHIT_MIR_VISITOR_H
