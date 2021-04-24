#include <graphit/midend/swarm_opt_lower.h>

namespace graphit {

void SwarmOptLower::lower(void) {
  for (auto func: mir_context_->getFunctionList()) {
    CoarseningAttacher coarsening(mir_context_);
    HintAttacher hint(mir_context_);
    func->accept(&coarsening);
    func->accept(&hint);
  }
}

void SwarmOptLower::CoarseningAttacher::visit(mir::PushEdgeSetApplyExpr::Ptr esae) {
  if (esae->hasApplySchedule()) {
    auto apply_schedule = esae->getApplySchedule();
    if (!apply_schedule->isComposite()) {
      auto applied_simple_schedule = apply_schedule->self<fir::swarm_schedule::SimpleSwarmSchedule>();
      if (applied_simple_schedule->enable_coarsening == fir::swarm_schedule::SimpleSwarmSchedule::CoarseningEnabled::COARSENING_ENABLED) {
        mir::DivExpr::Ptr default_coarsening_expr = std::make_shared<mir::DivExpr>();
	auto constant_cache_line = std::make_shared<mir::VarExpr>();
	auto scalar_type = std::make_shared<mir::ScalarType>();
	scalar_type->type = mir::ScalarType::Type::INT;
	constant_cache_line->var = mir::Var("SWARM_CACHE_LINE", scalar_type);

	mir::Call::Ptr sizeof_call = std::make_shared<mir::Call>();
	mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(esae->target);
	
	auto graph_type = mir_context_->getEdgesetType(var_expr->var.getName());
	mir::Type::Ptr var_type;
	if (mir::isa<mir::EdgeSetType>(graph_type) && mir::to<mir::EdgeSetType>(graph_type)->weight_type != nullptr) {
	  var_type = mir::to<mir::EdgeSetType>(graph_type)->weight_type;
	} else {
	  var_type = scalar_type;
	}
	sizeof_call->name = "builtin_sizeOf";
	sizeof_call->generic_type = var_type;

	default_coarsening_expr->lhs = constant_cache_line;// CONSTANT SWARM_CACHE_LINE
        default_coarsening_expr->rhs = sizeof_call;

	esae->setMetadata<mir::Expr::Ptr>("swarm_coarsen_expr", default_coarsening_expr);
      } 
    }
  }
}


void SwarmOptLower::HintAttacher::visit(mir::PushEdgeSetApplyExpr::Ptr esae) {
  if (esae->hasApplySchedule()) {
    auto apply_schedule = esae->getApplySchedule();
    if (!apply_schedule->isComposite()) {
      auto applied_simple_schedule = apply_schedule->self<fir::swarm_schedule::SimpleSwarmSchedule>();
      if (applied_simple_schedule->enable_hints == fir::swarm_schedule::SimpleSwarmSchedule::HintsEnabled::HINTS_ENABLED) {
	  std::string possible_hint = "";
	  if (esae->tracking_field != "") {
	    possible_hint = esae->tracking_field;
	  } else {
	    HintCandidateFinder finder(mir_context_);
	    std::string target_func = esae->input_function_name;
	    mir::FuncDecl::Ptr func_ptr = mir_context_->functions_map_[target_func];
	    func_ptr->accept(&finder);
	    if (finder.tensor_found) {
	      possible_hint = finder.tensor_name;
	    }
	  }
	  if (possible_hint == "") return;

	  mir::TensorArrayReadExpr::Ptr tare = std::make_shared<mir::TensorArrayReadExpr>();
	  mir::VarExpr::Ptr target_expr = std::make_shared<mir::VarExpr>();
	  mir::VectorType::Ptr vector_type = std::make_shared<mir::VectorType>();
	  mir::Var target_var = mir::Var(possible_hint, vector_type);
	  target_expr->var = target_var;
	  tare->target = target_expr; 
	  esae->setMetadata<mir::Expr::Ptr>("spatial_hint", tare);
      }
    }
  }
}

void SwarmOptLower::HintCandidateFinder::visit(mir::AssignStmt::Ptr assign_stmt) {
  if (mir::isa<mir::TensorArrayReadExpr>(assign_stmt->lhs)) {
    if (mir::isa<mir::VarExpr>(assign_stmt->lhs)) {
	    mir::VarExpr::Ptr target_expr = mir::to<mir::VarExpr>(assign_stmt->lhs);
	    if (!tensor_found) {
		    tensor_name = target_expr->var.getName();
		    tensor_found = true;
	    } else if (tensor_found && tensor_name != target_expr->var.getName()) {
		    tensor_found = false;
	    }
    }
  }
}

void SwarmOptLower::HintCandidateFinder::visit(mir::ReduceStmt::Ptr reduce_stmt) {
  if (mir::isa<mir::TensorArrayReadExpr>(reduce_stmt->lhs)) {
    auto target_expr = mir::to<mir::TensorArrayReadExpr>(reduce_stmt->lhs)->target;
    if (mir::isa<mir::VarExpr>(target_expr)) {
	    mir::VarExpr::Ptr target_var_expr = mir::to<mir::VarExpr>(target_expr);
	    if (!tensor_found) {
		    tensor_name = target_var_expr->var.getName();
		    tensor_found = true;
	    } else if (tensor_found && tensor_name != target_var_expr->var.getName()) {
		    tensor_found = false;
	    }
    }
  }
}

}
