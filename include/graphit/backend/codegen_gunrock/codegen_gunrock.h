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
			CodeGenGunrock(std::ostream &input_oss, MIRContext *mir_context) : oss(input_oss), mir_context_(mir_context), indent_value(0), current_context(mir::FuncDecl::CONTEXT_HOST){

			}
			int genGunrockCode(void);

		protected:
			int genIncludeStmts(void);
			int genEdgeSets(void);
			int genVertexSets(void);
			int genPropertyArrayDecl(mir::VarDecl::Ptr);
			int genPropertyArrayAlloca(mir::VarDecl::Ptr);
			int fillLambdaBody(mir::FuncDecl::Ptr, std::vector<std::string>);
			std::string getAllGlobals(void);



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
			virtual void visit(mir::VertexSetAllocExpr::Ptr);
			virtual void visit(mir::MulExpr::Ptr);
			virtual void visit(mir::StmtBlock::Ptr);
			virtual void visit(mir::PushEdgeSetApplyExpr::Ptr);
			virtual void visit(mir::FloatLiteral::Ptr);
			virtual void visit(mir::DivExpr::Ptr);
			virtual void visit(mir::SubExpr::Ptr);
			virtual void visit(mir::VertexSetWhereExpr::Ptr);
			


			// Defaults - 
			DEFAULT(CompareAndSwapStmt)
			DEFAULT(EdgeSetLoadExpr)
			DEFAULT(EdgeSetType)
			DEFAULT(EdgeSetWhereExpr)
			DEFAULT(ListAllocExpr)
			DEFAULT(ListType)
			DEFAULT(NegExpr)
			DEFAULT(PullEdgeSetApplyExpr)
			DEFAULT(StructTypeDecl)
			DEFAULT(TensorReadExpr)
			DEFAULT(TensorStructReadExpr)
			DEFAULT(VectorType)


			int indent(void);
			int dedent(void);
			void printIndent(void);

		private:
			MIRContext * mir_context_;
			std::ostream &oss;
			int indent_value;
			enum mir::FuncDecl::function_context current_context;
	};
	class ExtractReadWriteSet : public mir::MIRVisitor {
		public: 
			ExtractReadWriteSet() : read_set(read_set_), write_set(write_set_){
			}
			const std::vector<mir::Var> &read_set;
			const std::vector<mir::Var> &write_set;
			std::vector<mir::Var> getReadSet(void);
			std::vector<mir::Var> getWriteSet(void);
			
		protected:
			virtual void visit(mir::TensorArrayReadExpr::Ptr);
			virtual void visit(mir::AssignStmt::Ptr);
			virtual void visit(mir::StmtBlock::Ptr);
		private:
			void add_read(mir::Var);
			void add_write(mir::Var);
			std::vector<mir::Var> read_set_;
			std::vector<mir::Var> write_set_;
	};
}


#endif
