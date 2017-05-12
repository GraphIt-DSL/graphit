//
// Created by Yunming Zhang on 5/12/17.
//

#ifndef GRAPHIT_MIR_REWRITER_H
#define GRAPHIT_MIR_REWRITER_H

#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir.h>

namespace graphit {
    namespace mir {

        struct MIRRewriter : public MIRVisitor {
            virtual void visit(std::shared_ptr<Stmt> op) {node = op;};

            virtual void visit(std::shared_ptr<ForStmt>);

            virtual void visit(std::shared_ptr<ForDomain>);

            virtual void visit(std::shared_ptr<AssignStmt>);

            virtual void visit(std::shared_ptr<PrintStmt>);

            virtual void visit(std::shared_ptr<ExprStmt>);

            virtual void visit(std::shared_ptr<StmtBlock>);

            virtual void visit(std::shared_ptr<Expr>);

            virtual void visit(std::shared_ptr<Call>);

            virtual void visit(std::shared_ptr<ApplyExpr>);

            virtual void visit(std::shared_ptr<TensorReadExpr>);

            virtual void visit(std::shared_ptr<StringLiteral> op) {node = op;};

            virtual void visit(std::shared_ptr<FloatLiteral> op) {node = op;};

            virtual void visit(std::shared_ptr<IntLiteral> op) {node = op;} //leaf FIR nodes need no recursive calls
            virtual void visit(std::shared_ptr<VertexSetAllocExpr>);

            virtual void visit(std::shared_ptr<VarExpr> op) {node = op;};

            virtual void visit(std::shared_ptr<AddExpr>);

            virtual void visit(std::shared_ptr<SubExpr>);

            virtual void visit(std::shared_ptr<MulExpr>);

            virtual void visit(std::shared_ptr<DivExpr>);

            virtual void visit(std::shared_ptr<Type>) {};

            virtual void visit(std::shared_ptr<ScalarType> op) {node = op;};

            virtual void visit(std::shared_ptr<StructTypeDecl>);

            virtual void visit(std::shared_ptr<VarDecl>);

            virtual void visit(std::shared_ptr<IdentDecl> op) {node = op;};

            virtual void visit(std::shared_ptr<FuncDecl>);

            virtual void visit(std::shared_ptr<ElementType> op) {node = op;};

            virtual void visit(std::shared_ptr<VertexSetType>);

            virtual void visit(std::shared_ptr<EdgeSetType>);

            virtual void visit(std::shared_ptr<VectorType>);


            template <typename T = Program>
            std::shared_ptr<T> rewrite(std::shared_ptr<T> ptr) {
                auto tmp = node;

                ptr->accept(this);
                auto ret = std::static_pointer_cast<T>(node);

                node = tmp;

                return ret;
            }

            virtual void visitBinaryExpr(std::shared_ptr<BinaryExpr>);

        protected:
            MIRNode::Ptr node;
        };
    }
}

#endif //GRAPHIT_MIR_REWRITER_H
