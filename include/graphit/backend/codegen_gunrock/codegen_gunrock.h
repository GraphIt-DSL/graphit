#ifndef GRAPHIT_CODEGEN_GUNROCK_H
#define GRAPHIT_CODEGEN_GUNROCK_H


#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>

#include <iostream>
#include <sstream>


#define DEFAULT(T) virtual void visit(mir::T::Ptr x) { std::cerr << "Is a " #T "\n";}



namespace graphit {
	class CodeGenGunrock : mir::MIRVisitor {
		public:
			CodeGenGunrock(std::ostream &input_oss, MIRContext *mir_context) : oss(input_oss), mir_context_(mir_context) {
			}
			int genGunrockCode(void);

		protected:
			int genIncludeStmts(void);
			int genEdgeSets(void);
			int genPropertyArrayDecl(mir::VarDecl::Ptr);
			int genPropertyArrayAlloca(mir::VarDecl::Ptr);



			virtual void visit(mir::FuncDecl::Ptr);
			virtual void visit(mir::ScalarType::Ptr);
			virtual void visit(mir::Call::Ptr);
			virtual void visit(mir::VarExpr::Ptr);
			virtual void visit(mir::StringLiteral::Ptr);
			virtual void visit(mir::TensorArrayReadExpr::Ptr);
			virtual void visit(mir::IntLiteral::Ptr);
			virtual void visit(mir::ExprStmt::Ptr);
			virtual void visit(mir::VertexSetApplyExpr::Ptr);
			virtual void visit(mir::VarDecl::Ptr);
			virtual void visit(mir::ElementType::Ptr);
			virtual void visit(mir::AssignStmt::Ptr);
			virtual void visit(mir::ReduceStmt::Ptr);
			virtual void visit(mir::AddExpr::Ptr);
			virtual void visit(mir::ForStmt::Ptr);
			virtual void visit(mir::VertexSetType::Ptr);
			virtual void visit(mir::WhileStmt::Ptr);
			virtual void visit(mir::PrintStmt::Ptr);
			virtual void visit(mir::EqExpr::Ptr);
			virtual void visit(mir::IfStmt::Ptr);
			virtual void visit(mir::BreakStmt::Ptr);
			virtual void visit(mir::BoolLiteral::Ptr);


			// Defaults - 
			DEFAULT(CompareAndSwapStmt)
			DEFAULT(DivExpr)
			DEFAULT(EdgeSetLoadExpr)
			DEFAULT(EdgeSetType)
			DEFAULT(EdgeSetWhereExpr)
			DEFAULT(FloatLiteral)
			DEFAULT(ListAllocExpr)
			DEFAULT(ListType)
			DEFAULT(MulExpr)
			DEFAULT(NegExpr)
			DEFAULT(PullEdgeSetApplyExpr)
			DEFAULT(PushEdgeSetApplyExpr)
			DEFAULT(StructTypeDecl)
			DEFAULT(SubExpr)
			DEFAULT(TensorReadExpr)
			DEFAULT(TensorStructReadExpr)
			DEFAULT(VectorType)
			DEFAULT(VertexSetAllocExpr)
			DEFAULT(VertexSetWhereExpr)


			int indent(void);
			int dedent(void);
			void printIndent(void);

		private:
			MIRContext * mir_context_;
			std::ostream &oss;
			int indent_value;
	};
}


#endif
