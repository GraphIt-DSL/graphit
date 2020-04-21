#include "graphit/backend/codegen_gpu/assign_function_context.h"


namespace graphit {
int AssignFunctionContext::assign_function_context(void) {
	const std::vector<mir::FuncDecl::Ptr> &functions = mir_context_->getFunctionList();
	for (auto it = functions.begin(); it != functions.end(); it++)
		it->get()->accept(this);	
	for (auto stmt: mir_context_->field_vector_init_stmts)
		stmt->accept(this);
	
}
void AssignFunctionContext::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
	if (mir_context_->isFunction(pesae->input_function_name))
		mir_context_->getFunction(pesae->input_function_name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(pesae->from_func))
		mir_context_->getFunction(pesae->from_func)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(pesae->to_func))
		mir_context_->getFunction(pesae->to_func)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr pesae) {
	if (mir_context_->isFunction(pesae->input_function_name))
		mir_context_->getFunction(pesae->input_function_name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(pesae->from_func))
		mir_context_->getFunction(pesae->from_func)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(pesae->to_func))
		mir_context_->getFunction(pesae->to_func)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
	if (mir_context_->isFunction(pesae->input_function_name))
		mir_context_->getFunction(pesae->input_function_name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(pesae->from_func))
		mir_context_->getFunction(pesae->from_func)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(pesae->to_func))
		mir_context_->getFunction(pesae->to_func)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::VertexSetApplyExpr::Ptr vsae) {
	if (mir_context_->isFunction(vsae->input_function_name))
		mir_context_->getFunction(vsae->input_function_name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::VertexSetWhereExpr::Ptr vswe) {
	if (mir_context_->isFunction(vswe->input_func))
		mir_context_->getFunction(vswe->input_func)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
}
