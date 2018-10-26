#include "graphit/backend/codegen_gunrock/assign_function_context.h"

namespace graphit {
	int AssignFunctionContext::assign_function_context(void) {
		for (auto stmt: mir_context_->field_vector_init_stmts) {
			stmt->accept(this);
		}
		const std::vector<mir::FuncDecl::Ptr> &functions = mir_context_->getFunctionList();
		for (auto it = functions.begin(); it != functions.end(); it++) {
			it->get()->accept(this);	
		}	
		return 0;
	}	
	void AssignFunctionContext::visit(mir::VertexSetApplyExpr::Ptr vsae) {
		if (mir_context_->isFunction(vsae->input_function_name))
			mir_context_->getFunction(vsae->input_function_name)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;	
	}

	void AssignFunctionContext::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
		if(mir_context_->isFunction(pesae->input_function_name))
			mir_context_->getFunction(pesae->input_function_name)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;
		if(mir_context_->isFunction(pesae->from_func))
			mir_context_->getFunction(pesae->from_func)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;
		if(mir_context_->isFunction(pesae->to_func))
			mir_context_->getFunction(pesae->to_func)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;
	}
	void AssignFunctionContext::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
		if(mir_context_->isFunction(pesae->input_function_name))
			mir_context_->getFunction(pesae->input_function_name)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;
		if(mir_context_->isFunction(pesae->from_func))
			mir_context_->getFunction(pesae->from_func)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;
		if(mir_context_->isFunction(pesae->to_func))
			mir_context_->getFunction(pesae->to_func)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;
	}
	void AssignFunctionContext::visit(mir::VertexSetWhereExpr::Ptr vswe) {
		if(mir_context_->isFunction(vswe->input_func))
			mir_context_->getFunction(vswe->input_func)->realized_context |= mir::FuncDecl::CONTEXT_DEVICE;
	}
}

