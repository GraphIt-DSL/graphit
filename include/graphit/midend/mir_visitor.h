//
// Created by Yunming Zhang on 2/10/17.
//

#ifndef GRAPHIT_MIR_VISITOR_H
#define GRAPHIT_MIR_VISITOR_H

#include <memory>

namespace graphit {
    namespace mir {

        struct MIRNode;

        struct Program;
        struct Stmt;

        struct WhileStmt;
        struct ForStmt;
        struct ForDomain;
        struct ExprStmt;
        struct PrintStmt;
        struct AssignStmt;
        struct StmtBlock;
        struct Expr;

        struct BoolLiteral;
        struct StringLiteral;
        struct FloatLiteral;
        struct IntLiteral;
        struct Call;

        struct VertexSetApplyExpr;
        struct EdgeSetApplyExpr;
        struct VertexSetWhereExpr;
        struct EdgeSetWhereExpr;

        struct TensorReadExpr;
        struct TensorArrayReadExpr;
        struct TensorStructReadExpr;

        struct LoadExpr;
        struct VertexSetAllocExpr;
        struct VarExpr;
        struct MulExpr;
        struct DivExpr;
        struct AddExpr;
        struct SubExpr;
        struct BinaryExpr;
        struct Type;
        struct ScalarType;

        struct NegExpr;
        struct NaryExpr;
        struct EqExpr;


        struct StructTypeDecl;
        struct VarDecl;
        struct IdentDecl;
        struct FuncDecl;


        struct ElementType;
        struct VertexSetType;
        struct EdgeSetType;
        struct VectorType;

        struct MIRVisitor {
            virtual void visit(std::shared_ptr<Stmt>){};


            virtual void visit(std::shared_ptr<ForStmt>);
            virtual void visit(std::shared_ptr<WhileStmt>);


            virtual void visit(std::shared_ptr<ForDomain>);
            virtual void visit(std::shared_ptr<AssignStmt>);
            virtual void visit(std::shared_ptr<PrintStmt>);
            virtual void visit(std::shared_ptr<ExprStmt>);
            virtual void visit(std::shared_ptr<StmtBlock>);
            virtual void visit(std::shared_ptr<Expr>);
            virtual void visit(std::shared_ptr<Call>);

            virtual void visit(std::shared_ptr<VertexSetApplyExpr>);
            virtual void visit(std::shared_ptr<EdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<VertexSetWhereExpr>);
            virtual void visit(std::shared_ptr<EdgeSetWhereExpr>);


            virtual void visit(std::shared_ptr<TensorReadExpr>);
            virtual void visit(std::shared_ptr<TensorArrayReadExpr>);
            virtual void visit(std::shared_ptr<TensorStructReadExpr>);

            virtual void visit(std::shared_ptr<BoolLiteral>){};
            virtual void visit(std::shared_ptr<StringLiteral>){};
            virtual void visit(std::shared_ptr<FloatLiteral>){};
            virtual void visit(std::shared_ptr<IntLiteral> op) {} //leaf FIR nodes need no recursive calls
            virtual void visit(std::shared_ptr<VertexSetAllocExpr>);
            virtual void visit(std::shared_ptr<VarExpr>){};

            virtual void visit(std::shared_ptr<NegExpr>);
            virtual void visit(std::shared_ptr<EqExpr>);
            virtual void visit(std::shared_ptr<AddExpr>);
            virtual void visit(std::shared_ptr<SubExpr>);
            virtual void visit(std::shared_ptr<MulExpr>);
            virtual void visit(std::shared_ptr<DivExpr>);
            virtual void visit(std::shared_ptr<Type>){};
            virtual void visit(std::shared_ptr<ScalarType>){};
            virtual void visit(std::shared_ptr<StructTypeDecl>);
            virtual void visit(std::shared_ptr<VarDecl>);
            virtual void visit(std::shared_ptr<IdentDecl>){};
            virtual void visit(std::shared_ptr<FuncDecl>);
            virtual void visit(std::shared_ptr<ElementType>){};
            virtual void visit(std::shared_ptr<VertexSetType>);
            virtual void visit(std::shared_ptr<EdgeSetType>);
            virtual void visit(std::shared_ptr<VectorType>);


            virtual void visitBinaryExpr(std::shared_ptr<BinaryExpr>);
            virtual void visitNaryExpr(std::shared_ptr<NaryExpr>);

        };
    }
}


#endif //GRAPHIT_MIR_VISITOR_H
