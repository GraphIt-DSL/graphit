//
// Created by Yunming Zhang on 5/12/17.
//

#ifndef GRAPHIT_MIR_REWRITER_H
#define GRAPHIT_MIR_REWRITER_H

#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir.h>
#include <assert.h>
#include <graphit/midend/label_scope.h>

namespace graphit {
    namespace mir {

        struct MIRRewriter : public MIRVisitor {
            virtual void visit(std::shared_ptr<Stmt> op) { node = op; };

            virtual void visit(std::shared_ptr<NameNode>);


            virtual void visit(std::shared_ptr<ForStmt>);

            virtual void visit(std::shared_ptr<WhileStmt>);

            virtual void visit(std::shared_ptr<IfStmt>);


            virtual void visit(std::shared_ptr<ForDomain>);

            virtual void visit(std::shared_ptr<AssignStmt>);

            virtual void visit(std::shared_ptr<ReduceStmt>);

            virtual void visit(std::shared_ptr<CompareAndSwapStmt>);


            virtual void visit(std::shared_ptr<PrintStmt>);

            virtual void visit(std::shared_ptr<BreakStmt> stmt) { node = stmt; };

            virtual void visit(std::shared_ptr<ExprStmt>);

            virtual void visit(std::shared_ptr<StmtBlock>);

            virtual void visit(std::shared_ptr<Expr>);

            virtual void visit(std::shared_ptr<Call>);

            virtual void visit(std::shared_ptr<VertexSetApplyExpr>);

            virtual void visit(std::shared_ptr<EdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<PushEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<PullEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<HybridDenseForwardEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<HybridDenseEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<VertexSetWhereExpr>);


            virtual void visit(std::shared_ptr<TensorReadExpr>);

            virtual void visit(std::shared_ptr<TensorStructReadExpr>);

            virtual void visit(std::shared_ptr<TensorArrayReadExpr>);

            virtual void visit(std::shared_ptr<BoolLiteral> op) { node = op; };

            virtual void visit(std::shared_ptr<StringLiteral> op) { node = op; };

            virtual void visit(std::shared_ptr<FloatLiteral> op) { node = op; };

            virtual void visit(std::shared_ptr<IntLiteral> op) { node = op; } //leaf FIR nodes need no recursive calls

            virtual void visit(std::shared_ptr<VertexSetAllocExpr>);

            virtual void visit(std::shared_ptr<ListAllocExpr>);

            virtual void visit(std::shared_ptr<VarExpr> op) { node = op; };

            virtual void visit(std::shared_ptr<EdgeSetLoadExpr> op);

            virtual void visit(std::shared_ptr<IntersectionExpr> op);

            virtual void visit(std::shared_ptr<IntersectNeighborExpr> op);

            virtual void visit(std::shared_ptr<NegExpr>);

            virtual void visit(std::shared_ptr<EqExpr>);


            virtual void visit(std::shared_ptr<AndExpr>);

            virtual void visit(std::shared_ptr<OrExpr>);

            virtual void visit(std::shared_ptr<XorExpr>);

            virtual void visit(std::shared_ptr<NotExpr>);
            

            virtual void visit(std::shared_ptr<AddExpr>);

            virtual void visit(std::shared_ptr<SubExpr>);

            virtual void visit(std::shared_ptr<MulExpr>);

            virtual void visit(std::shared_ptr<DivExpr>);

            virtual void visit(std::shared_ptr<Type>) {};

            virtual void visit(std::shared_ptr<ScalarType> op) { node = op; };

            virtual void visit(std::shared_ptr<StructTypeDecl>);

            virtual void visit(std::shared_ptr<VarDecl>);

            virtual void visit(std::shared_ptr<IdentDecl> op) { node = op; };

            virtual void visit(std::shared_ptr<FuncDecl>);

            virtual void visit(std::shared_ptr<ElementType> op) { node = op; };

            virtual void visit(std::shared_ptr<VertexSetType>);

            virtual void visit(std::shared_ptr<ListType>);

            virtual void visit(std::shared_ptr<EdgeSetType>);

            virtual void visit(std::shared_ptr<VectorType>);

            virtual void visit(std::shared_ptr<VectorAllocExpr>);


            // OG Additions
            virtual void visit(std::shared_ptr<UpdatePriorityEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<UpdatePriorityExternVertexSetApplyExpr>);

            virtual void visit(std::shared_ptr<PriorityQueueType>);

            virtual void visit(std::shared_ptr<PriorityQueueAllocExpr>);

            virtual void visit(std::shared_ptr<UpdatePriorityUpdateBucketsCall>);

            virtual void visit(std::shared_ptr<UpdatePriorityExternCall>);

            virtual void visit(std::shared_ptr<OrderedProcessingOperator>);

            virtual void visit(std::shared_ptr<PriorityUpdateOperator>);

            virtual void visit(std::shared_ptr<PriorityUpdateOperatorMin>);

            virtual void visit(std::shared_ptr<PriorityUpdateOperatorSum>);

	    virtual void visit(std::shared_ptr<UpdatePriorityEdgeCountEdgeSetApplyExpr>);

            template<typename T = Program>
            std::shared_ptr<T> rewrite(std::shared_ptr<T> ptr) {
                auto tmp = node;

                ptr->accept(this);
                auto ret = std::static_pointer_cast<T>(node);
                assert(ret != nullptr);
                node = tmp;

                return ret;
            }

            virtual void visitBinaryExpr(std::shared_ptr<BinaryExpr>);

            virtual void visitNaryExpr(std::shared_ptr<NaryExpr>);

            void rewrite_call_args(Call::Ptr expr);
            void rewrite_priority_update_operator(PriorityUpdateOperator::Ptr expr);

        };
    }
}

#endif //GRAPHIT_MIR_REWRITER_H
