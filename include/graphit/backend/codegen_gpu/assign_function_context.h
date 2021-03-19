#ifndef ASSIGN_FUNCTION_CONTEXT_H
#define ASSIGN_FUNCTION CONTEXT_H

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>

#include <iostream>
#include <sstream>


namespace graphit {
class AssignFunctionContext : mir::MIRVisitor {
	public:
		AssignFunctionContext(MIRContext *mir_context) : mir_context_(mir_context) {
		}
		int assign_function_context(void);
	protected:
		void visit(mir::PushEdgeSetApplyExpr::Ptr);
		void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr);
		void visit(mir::PullEdgeSetApplyExpr::Ptr);
		void visit(mir::VertexSetApplyExpr::Ptr);
		void visit(mir::VertexSetWhereExpr::Ptr);
	private:
		MIRContext *mir_context_;
};
}


#endif
