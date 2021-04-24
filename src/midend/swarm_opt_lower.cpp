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
  return;
}

}
