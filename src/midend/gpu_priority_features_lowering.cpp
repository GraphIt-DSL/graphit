#include <graphit/midend/gpu_priority_features_lowering.h>

namespace graphit {

using fir::abstract_schedule::ScheduleObject;
using fir::gpu_schedule::SimpleGPUSchedule;

void GPUPriorityFeaturesLowering::lower(void) {
	EdgeSetApplyPriorityRewriter rewriter(mir_context_, schedule_);
	for (auto func: mir_context_->getFunctionList()) {
		rewriter.rewrite(func);
	}
}
void GPUPriorityFeaturesLowering::EdgeSetApplyPriorityRewriter::visit(mir::ExprStmt::Ptr expr_stmt) {
	if (expr_stmt->stmt_label != "") {
		label_scope_.scope(expr_stmt->stmt_label);
	}
	if (mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(expr_stmt->expr)) {
		mir::UpdatePriorityEdgeSetApplyExpr::Ptr upesae = mir::to<mir::UpdatePriorityEdgeSetApplyExpr>(expr_stmt->expr);
		mir::FuncDecl::Ptr udf = mir_context_->getFunction(upesae->input_function_name);
		UDFPriorityQueueFinder finder(mir_context_);
		udf->accept(&finder);
		mir::Var pq = finder.getPriorityQueue();

		mir::Var frontier(pq.getName() + ".frontier_", nullptr);

		mir::VarExpr::Ptr lhs = std::make_shared<mir::VarExpr>();
		lhs->var = frontier;

		mir::AssignStmt::Ptr assign = std::make_shared<mir::AssignStmt>();
		assign->lhs = lhs;
		assign->expr = expr_stmt->expr;
		node = assign;

		upesae->setMetadata<bool>("is_parallel", true);
		upesae->setMetadata<bool>("requires_output", true);
		upesae->setMetadata<mir::Var>("priority_queue_used", pq);
		mir::VarExpr::Ptr edgeset_expr = mir::to<mir::VarExpr>(upesae->target);
		mir::EdgeSetType::Ptr edgeset_type = mir::to<mir::EdgeSetType>(edgeset_expr->var.getType());
		assert(edgeset_type->vertex_element_type_list->size() == 2);
		if (edgeset_type->weight_type != nullptr) {
		    upesae->is_weighted = true;
		}
		// Now apply the schedule to the operator
		if (schedule_->backend_identifier == Schedule::BackendID::GPU) {
			if (upesae->hasApplySchedule()) {
                auto apply_schedule = upesae->getApplySchedule();
                if (apply_schedule->isa<SimpleGPUSchedule>()) {
                    mir_context_->delta_ = apply_schedule->self<SimpleGPUSchedule>()->delta;
                    if (apply_schedule->self<SimpleGPUSchedule>()->direction == SimpleGPUSchedule::direction_type::DIR_PULL) {
                        mir_context_->graphs_with_transpose[mir::to<mir::VarExpr>(upesae->target)->var.getName()] = true;
                    }
                } else {
                    assert(false && "Schedule applied to edgesetapply must be a Simple Schedule");
                }
			}
		}

		PriorityUpdateOperatorRewriter rewriter(mir_context_, upesae);
		rewriter.rewrite(udf);
		if (expr_stmt->stmt_label != "") {
			label_scope_.unscope();
		}
		return;
	}
	if (expr_stmt->stmt_label != "") {
		label_scope_.unscope();
	}
	mir::MIRRewriter::visit(expr_stmt);
	return;
}

void GPUPriorityFeaturesLowering::UDFPriorityQueueFinder::visit(mir::PriorityUpdateOperator::Ptr call) {
	if (mir::isa<mir::VarExpr>(call->args[0])) {
		insertVar(mir::to<mir::VarExpr>(call->args[0])->var);
	}
}
void GPUPriorityFeaturesLowering::UDFPriorityQueueFinder::visit(mir::PriorityUpdateOperatorMin::Ptr call) {
	mir::PriorityUpdateOperator::Ptr puo = call;
	visit(puo);
}
void GPUPriorityFeaturesLowering::UDFPriorityQueueFinder::visit(mir::PriorityUpdateOperatorSum::Ptr call) {
	mir::PriorityUpdateOperator::Ptr puo = call;
	visit(puo);
}
void GPUPriorityFeaturesLowering::UDFPriorityQueueFinder::visit(mir::Call::Ptr call) {
	if (call->name == "updatePriorityMin" || call->name == "UpdatePrioritySum") {
		if (mir::isa<mir::VarExpr>(call->args[0])) {
			insertVar(mir::to<mir::VarExpr>(call->args[0])->var);
		}
	}
}
void GPUPriorityFeaturesLowering::UDFPriorityQueueFinder::insertVar(mir::Var to_insert) {
	for (auto var: priority_queues_used) {
		if (var.getName() == to_insert.getName())
			return;
	}
	priority_queues_used.push_back(to_insert);
}
mir::Var GPUPriorityFeaturesLowering::UDFPriorityQueueFinder::getPriorityQueue(void) {
	assert(priority_queues_used.size() == 1 && "Exactly one priority queue must be used in the UDF supplied to UpdatePriorityEdgeSetApplyExpr");
	return priority_queues_used[0];
}
void GPUPriorityFeaturesLowering::PriorityUpdateOperatorRewriter::visit(mir::Call::Ptr call) {
	if (call->name == "updatePriorityMin") {
		mir::PriorityUpdateOperatorMin::Ptr update_op = std::make_shared<mir::PriorityUpdateOperatorMin>();
		update_op->priority_queue = call->args[0];
		update_op->destination_node_id = call->args[1];
		update_op->old_val = call->args[2];
		update_op->new_val = call->args[3];
		update_op->setMetadata<mir::UpdatePriorityEdgeSetApplyExpr::Ptr>("edgeset_apply_expr", puesae_);
		node = update_op;
	} else if (call->name == "updatePrioritySum") {
		mir::PriorityUpdateOperatorSum::Ptr update_op = std::make_shared<mir::PriorityUpdateOperatorSum>();
		update_op->priority_queue = call->args[0];
		update_op->destination_node_id = call->args[1];
		update_op->setMetadata<mir::Expr::Ptr>("delta", call->args[2]);
		update_op->setMetadata<mir::Expr::Ptr>("minimum_val", call->args[3]);
		update_op->setMetadata<mir::UpdatePriorityEdgeSetApplyExpr::Ptr>("edgeset_apply_expr", puesae_);
		node = update_op;
	}
}
}
