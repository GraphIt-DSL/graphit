#include <graphit/midend/gpu_change_tracking_lower.h>

namespace graphit {
void GPUChangeTrackingLower::lower(void) {
	UdfArgChangeVisitor visitor(mir_context_);
	for (auto func: mir_context_->getFunctionList()) {
		func->accept(&visitor);
	}
}
void GPUChangeTrackingLower::UdfArgChangeVisitor::updateUdf(mir::FuncDecl::Ptr func_decl, mir::EdgeSetApplyExpr::Ptr esae) {
	if (esae->requires_output == false)
		return;
	//assert(func_decl->udf_tracking_var == "" && "Currently, each UDF can only be used by one EdgeSetApply");
	//func_decl->udf_tracking_var = esae->tracking_field;
	//func_decl->calling_edge_set_apply_expr = esae;

	mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(esae->target);	
	mir::EdgeSetType::Ptr edge_set_type = mir::to<mir::EdgeSetType>(var_expr->var.getType());
	mir::ElementType::Ptr element_type = (*(edge_set_type->vertex_element_type_list))[0];
	mir::VertexSetType::Ptr vertex_set_type = std::make_shared<mir::VertexSetType>();
	vertex_set_type->element = element_type;
	
	mir::Var new_arg("__output_frontier", vertex_set_type);
	func_decl->args.push_back(new_arg);
	
	// Now modify all the reduce stmts inside
	ReductionOpChangeVisitor visitor(mir_context_, esae->tracking_field, esae);
	func_decl->accept(&visitor);
}
void GPUChangeTrackingLower::UdfArgChangeVisitor::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
	mir::FuncDecl::Ptr func_decl = mir_context_->getFunction(pesae->input_function_name);	
	updateUdf(func_decl, pesae);
}
void GPUChangeTrackingLower::UdfArgChangeVisitor::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
	mir::FuncDecl::Ptr func_decl = mir_context_->getFunction(pesae->input_function_name);	
	updateUdf(func_decl, pesae);
}

void GPUChangeTrackingLower::ReductionOpChangeVisitor::visit(mir::StmtBlock::Ptr stmt_block) {
	std::vector<mir::Stmt::Ptr> new_stmts;
	for (auto stmt: *(stmt_block->stmts)) {
		stmt->accept(this);
		if (mir::isa<mir::ReduceStmt>(stmt)) {
			mir::ReduceStmt::Ptr reduce_stmt = mir::to<mir::ReduceStmt>(stmt);
			if (mir::isa<mir::TensorReadExpr>(reduce_stmt->lhs)) {
				mir::TensorReadExpr::Ptr tre = mir::to<mir::TensorReadExpr>(reduce_stmt->lhs);
				if (mir::isa<mir::VarExpr>(tre->target) && mir::to<mir::VarExpr>(tre->target)->var.getName() == udf_tracking_var) {
					std::string result_var_name = "result_var" + mir_context_->getUniqueNameCounterString();
					reduce_stmt->tracking_var_name_ = result_var_name;
					reduce_stmt->calling_edge_set_apply_expr = current_edge_set_apply_expr;
					
					mir::ScalarType::Ptr scalar_type = std::make_shared<mir::ScalarType>();
					scalar_type->type = mir::ScalarType::Type::BOOL;
					mir::BoolLiteral::Ptr bool_literal = std::make_shared<mir::BoolLiteral>();
					bool_literal->val = false;
					mir::VarDecl::Ptr decl_stmt = std::make_shared<mir::VarDecl>();
					decl_stmt->name = result_var_name;
					decl_stmt->type = scalar_type;
					decl_stmt->initVal = bool_literal;
					new_stmts.push_back(decl_stmt);
				}
			}
		} else if (mir::isa<mir::CompareAndSwapStmt>(stmt)) {
			mir::CompareAndSwapStmt::Ptr cas_stmt = mir::to<mir::CompareAndSwapStmt>(stmt);
			if (mir::isa<mir::TensorReadExpr>(cas_stmt->lhs)) {
				mir::TensorReadExpr::Ptr tre = mir::to<mir::TensorReadExpr>(cas_stmt->lhs);
				if (mir::isa<mir::VarExpr>(tre->target) && mir::to<mir::VarExpr>(tre->target)->var.getName() == udf_tracking_var) {
					std::string result_var_name = "result_var" + mir_context_->getUniqueNameCounterString();
					cas_stmt->tracking_var_ = result_var_name;
					cas_stmt->calling_edge_set_apply_expr = current_edge_set_apply_expr;
					
					mir::ScalarType::Ptr scalar_type = std::make_shared<mir::ScalarType>();
					scalar_type->type = mir::ScalarType::Type::BOOL;
					mir::BoolLiteral::Ptr bool_literal = std::make_shared<mir::BoolLiteral>();
					bool_literal->val = false;
					mir::VarDecl::Ptr decl_stmt = std::make_shared<mir::VarDecl>();
					decl_stmt->name = result_var_name;
					decl_stmt->type = scalar_type;
					decl_stmt->initVal = bool_literal;
					new_stmts.push_back(decl_stmt);
				}
				
			}		
		}
		new_stmts.push_back(stmt);
	}
	*(stmt_block->stmts) = new_stmts;
}

}
