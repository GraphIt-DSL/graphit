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
	if (pesae->input_function && mir_context_->isFunction(pesae->input_function->function_name->name))
		mir_context_->getFunction(pesae->input_function->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (pesae->from_func && mir_context_->isFunction(pesae->from_func->function_name->name))
		mir_context_->getFunction(pesae->from_func->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (pesae->to_func && mir_context_->isFunction(pesae->to_func->function_name->name))
		mir_context_->getFunction(pesae->to_func->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr pesae) {
	if (pesae->input_function && mir_context_->isFunction(pesae->input_function->function_name->name))
		mir_context_->getFunction(pesae->input_function->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (pesae->from_func && mir_context_->isFunction(pesae->from_func->function_name->name))
		mir_context_->getFunction(pesae->from_func->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (pesae->to_func && mir_context_->isFunction(pesae->to_func->function_name->name))
		mir_context_->getFunction(pesae->to_func->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
	if (pesae->input_function && mir_context_->isFunction(pesae->input_function->function_name->name))
		mir_context_->getFunction(pesae->input_function->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (pesae->from_func && mir_context_->isFunction(pesae->from_func->function_name->name))
		mir_context_->getFunction(pesae->from_func->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (pesae->to_func && mir_context_->isFunction(pesae->to_func->function_name->name))
		mir_context_->getFunction(pesae->to_func->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::VertexSetApplyExpr::Ptr vsae) {
	if (vsae->input_function && mir_context_->isFunction(vsae->input_function->function_name->name))
		mir_context_->getFunction(vsae->input_function->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::VertexSetWhereExpr::Ptr vswe) {
	if (vswe->input_func && mir_context_->isFunction(vswe->input_func->function_name->name))
		mir_context_->getFunction(vswe->input_func->function_name->name)->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
}
