#include "graphit/backend/codegen_gpu/assign_function_context.h"


namespace graphit {
int AssignFunctionContext::assign_function_context(void) {
	const std::vector<mir::FuncDecl::Ptr> &functions = mir_context_->getFunctionList();
	for (auto it = functions.begin(); it != functions.end(); it++)
		it->get()->accept(this);	
	for (auto stmt: mir_context_->field_vector_init_stmts)
		stmt->accept(this);
	
}


static std::string getNameFromFuncExpr(mir::FuncExpr::Ptr fe) {
	if (fe == nullptr)
		return "";
	if (fe->function_name == nullptr)
		return "";
	return fe->function_name->name;
}
void AssignFunctionContext::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->input_function)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->input_function))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->from_func)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->from_func))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->to_func)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->to_func))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr pesae) {
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->input_function)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->input_function))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->from_func)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->from_func))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->to_func)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->to_func))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->input_function)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->input_function))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->from_func)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->from_func))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
	if (mir_context_->isFunction(getNameFromFuncExpr(pesae->to_func)))
		mir_context_->getFunction(getNameFromFuncExpr(pesae->to_func))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::VertexSetApplyExpr::Ptr vsae) {
	if (mir_context_->isFunction(getNameFromFuncExpr(vsae->input_function)))
		mir_context_->getFunction(getNameFromFuncExpr(vsae->input_function))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
void AssignFunctionContext::visit(mir::VertexSetWhereExpr::Ptr vswe) {
	if (mir_context_->isFunction(getNameFromFuncExpr(vswe->input_func)))
		mir_context_->getFunction(getNameFromFuncExpr(vswe->input_func))->function_context = mir::FuncDecl::function_context_type::CONTEXT_DEVICE;
}
}
