//
// Created by Yunming Zhang on 2/10/17.
//

#ifndef GRAPHIT_MIR_VISITOR_H
#define GRAPHIT_MIR_VISITOR_H

//make sure not to include mir.h here

#include <memory>
#include <assert.h>
#include <graphit/midend/label_scope.h>


namespace graphit {
    namespace mir {

        class Var;
        struct MIRNode;

        struct Program;
        struct Stmt;

        struct WhileStmt;
        struct ForStmt;
        struct IfStmt;

        struct ForDomain;
        struct NameNode;
        struct ExprStmt;
        struct PrintStmt;
        struct BreakStmt;
        struct AssignStmt;
        struct ReduceStmt;
        struct CompareAndSwapStmt;

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

        struct PushEdgeSetApplyExpr;
        struct PullEdgeSetApplyExpr;
        struct HybridDenseEdgeSetApplyExpr;
        struct HybridDenseForwardEdgeSetApplyExpr;

        struct TensorReadExpr;
        struct TensorArrayReadExpr;
        struct TensorStructReadExpr;

        struct IntersectionExpr;
        struct IntersectNeighborExpr;

        struct LoadExpr;
        struct EdgeSetLoadExpr;

        struct VertexSetAllocExpr;
        struct ListAllocExpr;
        struct VectorAllocExpr;


        struct AndExpr;
        struct OrExpr;
        struct XorExpr;
        struct NotExpr;

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
        struct FuncExpr;


        struct ElementType;
        struct VertexSetType;
        struct ListType;
        struct EdgeSetType;
        struct VectorType;


        // OG Additions
        struct PriorityQueueType;
        struct PriorityQueueAllocExpr;
        struct UpdatePriorityEdgeSetApplyExpr;
        struct UpdatePriorityExternVertexSetApplyExpr;
        struct UpdatePriorityUpdateBucketsCall;
	    struct UpdatePriorityExternCall;

	    struct OrderedProcessingOperator;

	    struct PriorityUpdateOperator;
	    struct PriorityUpdateOperatorMin;
	    struct PriorityUpdateOperatorSum;
		struct UpdatePriorityEdgeCountEdgeSetApplyExpr;


        struct MIRVisitor {
            virtual void visit(Var*);

            virtual void visit(std::shared_ptr<Stmt>) {};

            virtual void visit(std::shared_ptr<NameNode>);

            virtual void visit(std::shared_ptr<ForStmt>);

            virtual void visit(std::shared_ptr<WhileStmt>);

            virtual void visit(std::shared_ptr<IfStmt>);


            virtual void visit(std::shared_ptr<ForDomain>);

            virtual void visit(std::shared_ptr<AssignStmt>);

            virtual void visit(std::shared_ptr<ReduceStmt>);

            virtual void visit(std::shared_ptr<CompareAndSwapStmt>);

            virtual void visit(std::shared_ptr<PrintStmt>);

            virtual void visit(std::shared_ptr<BreakStmt>) {};

            virtual void visit(std::shared_ptr<ExprStmt>);

            virtual void visit(std::shared_ptr<StmtBlock>);

            virtual void visit(std::shared_ptr<Expr>);

            virtual void visit(std::shared_ptr<Call>);

            virtual void visit(std::shared_ptr<VertexSetApplyExpr>);

            virtual void visit(std::shared_ptr<EdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<PushEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<PullEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<HybridDenseEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<HybridDenseForwardEdgeSetApplyExpr>);


            virtual void visit(std::shared_ptr<VertexSetWhereExpr>);

            virtual void visit(std::shared_ptr<EdgeSetWhereExpr>);


            virtual void visit(std::shared_ptr<TensorReadExpr>);

            virtual void visit(std::shared_ptr<TensorArrayReadExpr>);

            virtual void visit(std::shared_ptr<TensorStructReadExpr>);

            virtual void visit(std::shared_ptr<BoolLiteral>) {};

            virtual void visit(std::shared_ptr<StringLiteral>) {};

            virtual void visit(std::shared_ptr<FloatLiteral>) {};

            virtual void visit(std::shared_ptr<IntLiteral> op) {} //leaf FIR nodes need no recursive calls
            virtual void visit(std::shared_ptr<VertexSetAllocExpr>);

            virtual void visit(std::shared_ptr<VectorAllocExpr>);

            virtual void visit(std::shared_ptr<ListAllocExpr>);

            virtual void visit(std::shared_ptr<VarExpr>) {};

            virtual void visit(std::shared_ptr<IntersectionExpr>);
            virtual void visit(std::shared_ptr<IntersectNeighborExpr>);


            virtual void visit(std::shared_ptr<EdgeSetLoadExpr>);


            virtual void visit(std::shared_ptr<AndExpr>);
            virtual void visit(std::shared_ptr<OrExpr>);
            virtual void visit(std::shared_ptr<XorExpr>);
            virtual void visit(std::shared_ptr<NotExpr>);

            virtual void visit(std::shared_ptr<NegExpr>);

            virtual void visit(std::shared_ptr<EqExpr>);

            virtual void visit(std::shared_ptr<AddExpr>);

            virtual void visit(std::shared_ptr<SubExpr>);

            virtual void visit(std::shared_ptr<MulExpr>);

            virtual void visit(std::shared_ptr<DivExpr>);

            virtual void visit(std::shared_ptr<Type>) {};

            virtual void visit(std::shared_ptr<ScalarType>) {};

            virtual void visit(std::shared_ptr<StructTypeDecl>);

            virtual void visit(std::shared_ptr<VarDecl>);

            virtual void visit(std::shared_ptr<IdentDecl>) {};

            virtual void visit(std::shared_ptr<FuncDecl>);

            virtual void visit(std::shared_ptr<FuncExpr>);

            virtual void visit(std::shared_ptr<ElementType>) {};

            virtual void visit(std::shared_ptr<VertexSetType>);

            virtual void visit(std::shared_ptr<ListType>);

            virtual void visit(std::shared_ptr<EdgeSetType>);

            virtual void visit(std::shared_ptr<VectorType>);


            virtual void visitBinaryExpr(std::shared_ptr<BinaryExpr>);

            virtual void visitNaryExpr(std::shared_ptr<NaryExpr>);

            // OG Additions
            virtual void visit(std::shared_ptr<PriorityQueueType>);

            virtual void visit(std::shared_ptr<PriorityQueueAllocExpr>);

            virtual void visit(std::shared_ptr<UpdatePriorityEdgeSetApplyExpr>);

            virtual void visit(std::shared_ptr<UpdatePriorityExternVertexSetApplyExpr>);

            virtual void visit(std::shared_ptr<UpdatePriorityUpdateBucketsCall>);
	
            virtual void visit(std::shared_ptr<UpdatePriorityExternCall>);

            virtual void visit(std::shared_ptr<OrderedProcessingOperator>);

			virtual void visit(std::shared_ptr<PriorityUpdateOperator>);

			virtual void visit(std::shared_ptr<PriorityUpdateOperatorMin>);

			virtual void visit(std::shared_ptr<PriorityUpdateOperatorSum>);


            virtual void visit(std::shared_ptr<UpdatePriorityEdgeCountEdgeSetApplyExpr>);
	    

		protected:
            std::shared_ptr<MIRNode> node;
            LabelScope label_scope_;
            std::shared_ptr<FuncDecl> enclosing_func_decl_ = nullptr;

        };
    }
}


#endif //GRAPHIT_MIR_VISITOR_H
