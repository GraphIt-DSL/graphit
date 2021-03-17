#include <graphit/midend/swarm_priority_features_lowering.h>

namespace graphit {

void SwarmPriorityFeaturesLowering::lower(void) {
  for (auto func: mir_context_->getFunctionList()) {
    PrioFrontierFinderVisitor visitor;
    func->accept(&visitor);
  }
}

void SwarmPriorityFeaturesLowering::PrioFrontierFinderVisitor::visit(mir::WhileStmt::Ptr while_stmt) {
  auto stmt_block = while_stmt->body;

    if (mir::isa<mir::EqExpr>(while_stmt->cond)) {
      mir::EqExpr::Ptr cond_expr = mir::to<mir::EqExpr>(while_stmt->cond);
      if (mir::isa<mir::Call>(cond_expr->operands[0])) {
        mir::Call::Ptr first_operand_call = mir::to<mir::Call>(cond_expr->operands[0]);
        if (mir::isa<mir::VarExpr>(first_operand_call->args[0])){
          mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(first_operand_call->args[0]);
          if (mir::isa<mir::PriorityQueueType>(var_expr->var.getType())) {
	    WhileStmtPQVisitor pq_visitor;
	    pq_visitor.pq_name = var_expr->var.getName(); 
            stmt_block->accept(&pq_visitor);
	  }
        }
      }
    }
}

void SwarmPriorityFeaturesLowering::WhileStmtPQVisitor::visit(mir::StmtBlock::Ptr stmt_block) {
  for (int i = 0; i <stmt_block->stmts->size();i++) {
    mir::Stmt::Ptr this_stmt = (*(stmt_block->stmts))[i];
    to_delete = false;
    this_stmt->accept(this);
    if (to_delete) to_deletes.push_back(this_stmt);
  }
  std::vector<mir::Stmt::Ptr> new_stmts;
  for (int i = 0; i < stmt_block->stmts->size(); i++) {
    mir::Stmt::Ptr this_stmt = (*(stmt_block->stmts))[i];
    if (std::find(to_deletes.begin(), to_deletes.end(), this_stmt) == to_deletes.end()) {
      new_stmts.push_back(this_stmt);
    }
  }
  (*(stmt_block->stmts)) = new_stmts;
}

void SwarmPriorityFeaturesLowering::WhileStmtPQVisitor::visit(mir::VarDecl::Ptr var_decl) {
  if (var_decl->initVal != nullptr) {
    if (mir::isa<mir::Call>(var_decl->initVal)) {
      mir::Call::Ptr call = mir::to<mir::Call>(var_decl->initVal);
      auto call_arg = call->args[0];
      if (mir::isa<mir::VarExpr>(call_arg)) {
	auto var_expr = mir::to<mir::VarExpr>(call_arg);
	std::string arg_name = var_expr->var.getName();
        if (call->name == "dequeue_ready_set" && arg_name == pq_name) {
          temp_frontier_name = var_decl->name;
	  to_delete = true;
        }
      }
    }
  }
} 

void SwarmPriorityFeaturesLowering::WhileStmtPQVisitor::visit(mir::Call::Ptr call) {
  if (temp_frontier_name != "" && call->args.size() == 1) {
    if (mir::isa<mir::VarExpr>(call->args[0])) {
      auto var_expr = mir::to<mir::VarExpr>(call->args[0]);
      if (var_expr->var.getName() == temp_frontier_name) {
        to_delete = true;
      }
    }
  }
}

void SwarmPriorityFeaturesLowering::WhileStmtPQVisitor::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr esae) {
  if (temp_frontier_name != "" && esae->from_func == temp_frontier_name) {
    esae->from_func = pq_name;
  }
}

}
