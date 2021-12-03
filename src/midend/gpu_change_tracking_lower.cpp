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

	mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(esae->target);	
	mir::EdgeSetType::Ptr edge_set_type = mir::to<mir::EdgeSetType>(var_expr->var.getType());
	mir::ElementType::Ptr element_type = (*(edge_set_type->vertex_element_type_list))[0];
	mir::VertexSetType::Ptr vertex_set_type = std::make_shared<mir::VertexSetType>();
	vertex_set_type->element = element_type;
	
	mir::Var new_arg("__output_frontier", vertex_set_type);
	func_decl->args.push_back(new_arg);
	
	// Now modify all the reduce stmts inside
	ReductionOpChangeVisitor visitor(mir_context_, esae->tracking_field, esae, vertex_set_type);
	func_decl->accept(&visitor);
}
void GPUChangeTrackingLower::UdfArgChangeVisitor::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
	mir::FuncDecl::Ptr func_decl = mir_context_->getFunction(pesae->input_function->function_name->name);	
	updateUdf(func_decl, pesae);
}
void GPUChangeTrackingLower::UdfArgChangeVisitor::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr pesae) {
	mir::FuncDecl::Ptr func_decl = mir_context_->getFunction(pesae->input_function->function_name->name);	
	updateUdf(func_decl, pesae);
}
void GPUChangeTrackingLower::UdfArgChangeVisitor::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
	mir::FuncDecl::Ptr func_decl = mir_context_->getFunction(pesae->input_function->function_name->name);	
	updateUdf(func_decl, pesae);
}

void GPUChangeTrackingLower::ReductionOpChangeVisitor::visit(mir::StmtBlock::Ptr stmt_block) {
	std::vector<mir::Stmt::Ptr> new_stmts;
	for (auto stmt: *(stmt_block->stmts)) {
		stmt->accept(this);
		bool stmt_added = false;
		if (mir::isa<mir::ReduceStmt>(stmt)) {
			mir::ReduceStmt::Ptr reduce_stmt = mir::to<mir::ReduceStmt>(stmt);
			if (mir::isa<mir::TensorReadExpr>(reduce_stmt->lhs)) {
				mir::TensorReadExpr::Ptr tre = mir::to<mir::TensorReadExpr>(reduce_stmt->lhs);
				if (mir::isa<mir::VarExpr>(tre->target) && mir::to<mir::VarExpr>(tre->target)->var.getName() == udf_tracking_var) {
					std::string result_var_name = "result_var" + mir_context_->getUniqueNameCounterString();
					reduce_stmt->tracking_var_name_ = result_var_name;
					
					mir::ScalarType::Ptr scalar_type = std::make_shared<mir::ScalarType>();
					scalar_type->type = mir::ScalarType::Type::BOOL;
					mir::BoolLiteral::Ptr bool_literal = std::make_shared<mir::BoolLiteral>();
					bool_literal->val = false;
					mir::VarDecl::Ptr decl_stmt = std::make_shared<mir::VarDecl>();
					decl_stmt->name = result_var_name;
					decl_stmt->type = scalar_type;
					decl_stmt->initVal = bool_literal;
					new_stmts.push_back(decl_stmt);
					new_stmts.push_back(stmt);

					// Now construct the conditional enqueue
					mir::Var tracking_var(result_var_name, scalar_type);
					mir::VarExpr::Ptr condition_expr = std::make_shared<mir::VarExpr>();
					condition_expr->var = tracking_var;
					mir::IfStmt::Ptr if_stmt = std::make_shared<mir::IfStmt>();
					if_stmt->cond = condition_expr;
					
					mir::StmtBlock::Ptr stmt_block = std::make_shared<mir::StmtBlock>();
					if_stmt->ifBody = stmt_block;
					
					mir::EnqueueVertex::Ptr enqueue_vertex = std::make_shared<mir::EnqueueVertex>();
					mir::Var frontier_var("__output_frontier", frontier_type);
					mir::VarExpr::Ptr frontier_expr = std::make_shared<mir::VarExpr>();
					frontier_expr->var = frontier_var;
					enqueue_vertex->vertex_id = tre->index;
					enqueue_vertex->vertex_frontier = frontier_expr;	
					enqueue_vertex->fused_dedup = current_edge_set_apply_expr->fused_dedup;
					enqueue_vertex->fused_dedup_perfect = current_edge_set_apply_expr->fused_dedup_perfect;
					if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::SPARSE;
					} else if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BOOLMAP) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::BOOLMAP;
					} else if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BITMAP) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::BITMAP;
					}
					stmt_block->insertStmtEnd(enqueue_vertex);
					if_stmt->elseBody = nullptr;
					new_stmts.push_back(if_stmt);
					stmt_added = true;
				}
			}
		} else if (mir::isa<mir::CompareAndSwapStmt>(stmt)) {
			mir::CompareAndSwapStmt::Ptr cas_stmt = mir::to<mir::CompareAndSwapStmt>(stmt);
			if (mir::isa<mir::TensorReadExpr>(cas_stmt->lhs)) {
				mir::TensorReadExpr::Ptr tre = mir::to<mir::TensorReadExpr>(cas_stmt->lhs);
				if (mir::isa<mir::VarExpr>(tre->target) && mir::to<mir::VarExpr>(tre->target)->var.getName() == udf_tracking_var) {
					std::string result_var_name = "result_var" + mir_context_->getUniqueNameCounterString();
					cas_stmt->tracking_var_ = result_var_name;
					
					mir::ScalarType::Ptr scalar_type = std::make_shared<mir::ScalarType>();
					scalar_type->type = mir::ScalarType::Type::BOOL;
					mir::BoolLiteral::Ptr bool_literal = std::make_shared<mir::BoolLiteral>();
					bool_literal->val = false;
					mir::VarDecl::Ptr decl_stmt = std::make_shared<mir::VarDecl>();
					decl_stmt->name = result_var_name;
					decl_stmt->type = scalar_type;
					decl_stmt->initVal = bool_literal;
					new_stmts.push_back(decl_stmt);
					new_stmts.push_back(stmt);

					// Now construct the conditional enqueue
					mir::Var tracking_var(result_var_name, scalar_type);
					mir::VarExpr::Ptr condition_expr = std::make_shared<mir::VarExpr>();
					condition_expr->var = tracking_var;
					mir::IfStmt::Ptr if_stmt = std::make_shared<mir::IfStmt>();
					if_stmt->cond = condition_expr;
					
					mir::StmtBlock::Ptr stmt_block = std::make_shared<mir::StmtBlock>();
					if_stmt->ifBody = stmt_block;
					
					mir::EnqueueVertex::Ptr enqueue_vertex = std::make_shared<mir::EnqueueVertex>();
					mir::Var frontier_var("__output_frontier", frontier_type);
					mir::VarExpr::Ptr frontier_expr = std::make_shared<mir::VarExpr>();
					frontier_expr->var = frontier_var;
					enqueue_vertex->vertex_id = tre->index;
					enqueue_vertex->vertex_frontier = frontier_expr;	
					enqueue_vertex->fused_dedup = current_edge_set_apply_expr->fused_dedup;
					enqueue_vertex->fused_dedup_perfect = current_edge_set_apply_expr->fused_dedup_perfect;
					if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::SPARSE;
					} else if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BOOLMAP) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::BOOLMAP;
					} else if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BITMAP) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::BITMAP;
					}
					stmt_block->insertStmtEnd(enqueue_vertex);
					if_stmt->elseBody = nullptr;
					new_stmts.push_back(if_stmt);
					stmt_added = true;
				}
				
			}		
		} else if (mir::isa<mir::AssignStmt>(stmt)) {
			mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(stmt);
			if (mir::isa<mir::TensorReadExpr>(assign_stmt->lhs)) {
				mir::TensorReadExpr::Ptr tre = mir::to<mir::TensorReadExpr>(assign_stmt->lhs);
				if (mir::isa<mir::VarExpr>(tre->target) && mir::to<mir::VarExpr>(tre->target)->var.getName() == udf_tracking_var) {
					new_stmts.push_back(stmt);
					mir::EnqueueVertex::Ptr enqueue_vertex = std::make_shared<mir::EnqueueVertex>();
					mir::Var frontier_var("__output_frontier", frontier_type);
					mir::VarExpr::Ptr frontier_expr = std::make_shared<mir::VarExpr>();
					frontier_expr->var = frontier_var;
					enqueue_vertex->vertex_id = tre->index;
					enqueue_vertex->vertex_frontier = frontier_expr;	
					enqueue_vertex->fused_dedup = current_edge_set_apply_expr->fused_dedup;
					enqueue_vertex->fused_dedup_perfect = current_edge_set_apply_expr->fused_dedup_perfect;
					if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::SPARSE;
					} else if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BOOLMAP) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::BOOLMAP;
					} else if (current_edge_set_apply_expr->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BITMAP) {
						enqueue_vertex->type = mir::EnqueueVertex::Type::BITMAP;
					}
					new_stmts.push_back(enqueue_vertex);
					stmt_added = true;
				}
			}
			
		}
		if (!stmt_added)
			new_stmts.push_back(stmt);
	}
	*(stmt_block->stmts) = new_stmts;
}

}
