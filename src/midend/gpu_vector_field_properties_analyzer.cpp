#include <graphit/midend/gpu_vector_field_properties_analyzer.h>

namespace graphit {
void GPUVectorFieldPropertiesAnalyzer::analyze(void) {
	ApplyExprVisitor visitor(mir_context_);
	for (auto func: mir_context_->getFunctionList()) {
		func->accept(&visitor);
	}	
}
void GPUVectorFieldPropertiesAnalyzer::ApplyExprVisitor::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
	// Push apply expression requires synchronization on src when using non vertex based load balance
	// Push apply expression always requires synchronization on dst
	std::unordered_set<std::string> idp_set;
	mir::FuncDecl::Ptr func = mir_context_->getFunction(pesae->input_function->function_name->name);

	std::string src_name = func->args[0].getName();
	std::string dst_name = func->args[1].getName();

	switch (pesae->applied_schedule.load_balancing) {
		case fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::VERTEX_BASED:
			idp_set.insert(src_name);
			break;
		default:
			break;	
	}	
	
	
	PropertyAnalyzingVisitor visitor(mir_context_, idp_set, func);
	func->accept(&visitor);
}
void GPUVectorFieldPropertiesAnalyzer::ApplyExprVisitor::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
	// Pull apply expression requires synchronization on dst when using non vertex based load balance
	// Pull apply expression always requires synchronization on src
	std::unordered_set<std::string> idp_set;
	mir::FuncDecl::Ptr func = mir_context_->getFunction(pesae->input_function->function_name->name);

	std::string src_name = func->args[0].getName();
	std::string dst_name = func->args[1].getName();

	switch (pesae->applied_schedule.load_balancing) {
		case fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::VERTEX_BASED:
			idp_set.insert(dst_name);
			break;
		default:
			break;	
	}	
	
	
	PropertyAnalyzingVisitor visitor(mir_context_, idp_set, func);
	func->accept(&visitor);
}

void GPUVectorFieldPropertiesAnalyzer::ApplyExprVisitor::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr pesae) {
	// UpdatePriority will function just like Push for now
	std::unordered_set<std::string> idp_set;
	mir::FuncDecl::Ptr func = mir_context_->getFunction(pesae->input_function->function_name->name);

	std::string src_name = func->args[0].getName();
	std::string dst_name = func->args[1].getName();

	switch (pesae->applied_schedule.load_balancing) {
		case fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::VERTEX_BASED:
			idp_set.insert(src_name);
			break;
		default:
			break;	
	}	
	
	
	PropertyAnalyzingVisitor visitor(mir_context_, idp_set, func);
	func->accept(&visitor);
	
}



bool GPUVectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::is_independent_index(mir::Expr::Ptr expr) {
	if (mir::isa<mir::VarExpr>(expr)) {
		mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(expr);
		if (independent_variables.count(var_expr->var.getName()) > 0) {
			return true;
		}
	}
	if (mir::isa<mir::AddExpr>(expr)) {
		mir::AddExpr::Ptr add_expr = mir::to<mir::AddExpr>(expr);
		if (mir::isa<mir::IntLiteral>(add_expr->lhs) && is_independent_index(add_expr->rhs))
			return true;
		if (mir::isa<mir::IntLiteral>(add_expr->rhs) && is_independent_index(add_expr->lhs))
			return true;
	}
	if (mir::isa<mir::MulExpr>(expr)) {
		mir::MulExpr::Ptr mul_expr = mir::to<mir::MulExpr>(expr);
		if (mir::isa<mir::IntLiteral>(mul_expr->lhs) && is_independent_index(mul_expr->rhs) && mir::to<mir::IntLiteral>(mul_expr->lhs)->val != 0)
			return true;
		if (mir::isa<mir::IntLiteral>(mul_expr->rhs) && is_independent_index(mul_expr->lhs) && mir::to<mir::IntLiteral>(mul_expr->rhs)->val != 0)
			return true;
	}

	return false;
	
}
void GPUVectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::TensorReadExpr::Ptr tre) {

	tre->index->accept(this);

	FieldVectorProperty property;
	property.read_write_type = FieldVectorProperty::ReadWriteType::READ_ONLY;
	if (is_independent_index(tre->index)) {
		property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
	} else {
		property.access_type_ = FieldVectorProperty::AccessType::SHARED;
	}
	tre->field_vector_prop_  = property;
	std::string target = tre->getTargetNameStr();
	enclosing_function->field_vector_properties_map_[target] = property;
}
void GPUVectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::AssignStmt::Ptr assign_stmt) {

	assign_stmt->expr->accept(this);
	
	if (!mir::isa<mir::TensorReadExpr>(assign_stmt->lhs))
		return;	


	mir::TensorReadExpr::Ptr tre = mir::to<mir::TensorReadExpr>(assign_stmt->lhs);
	tre->index->accept(this);
	FieldVectorProperty property;
	property.read_write_type = FieldVectorProperty::ReadWriteType::WRITE_ONLY;
	if (is_independent_index(tre->index)) {
		property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
	} else {
		property.access_type_ = FieldVectorProperty::AccessType::SHARED;
	}
	tre->field_vector_prop_ = property;
	std::string target = tre->getTargetNameStr();
	enclosing_function->field_vector_properties_map_[target] = property;	
}
void GPUVectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::ReduceStmt::Ptr reduce_stmt) {
	reduce_stmt->expr->accept(this);
	
	if (!mir::isa<mir::TensorReadExpr>(reduce_stmt->lhs))
		return;
	mir::TensorReadExpr::Ptr tre = mir::to<mir::TensorReadExpr>(reduce_stmt->lhs);
	tre->index->accept(this);
	FieldVectorProperty property;
	property.read_write_type = FieldVectorProperty::ReadWriteType::READ_AND_WRITE;
	if (is_independent_index(tre->index)) {
		property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
	} else {
		property.access_type_ = FieldVectorProperty::AccessType::SHARED;
	}
	tre->field_vector_prop_ = property;
	std::string target = tre->getTargetNameStr();
	enclosing_function->field_vector_properties_map_[target] = property;	
	
}
void GPUVectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::PriorityUpdateOperatorMin::Ptr puo) {
	mir::MIRVisitor::visit(puo);
	mir::Expr::Ptr index_expr = puo->destination_node_id;
	if (!is_independent_index(index_expr)) {
		puo->is_atomic = true;	
	}
}
}
