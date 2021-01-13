#include <graphit/midend/swarm_frontier_convert.h>

namespace graphit {
void SwarmFrontierConvert::analyze(void) {
  for (auto func: mir_context_->getFunctionList()) {
    func->accept(this);
  }
}

void SwarmFrontierConvert::visit(mir::WhileStmt::Ptr while_stmt) {
  while_stmt->setMetadata<bool>("swarm_frontier_convert", false);
  while_stmt->setMetadata<bool>("swarm_switch_convert", false);


  if (mir::isa<mir::EqExpr>(while_stmt->cond)) {
    mir::EqExpr::Ptr cond_expr = mir::to<mir::EqExpr>(while_stmt->cond);
    if (mir::isa<mir::Call>(cond_expr->operands[0])) {
      mir::Call::Ptr first_operand_call = mir::to<mir::Call>(cond_expr->operands[0]);
      if (mir::isa<mir::VarExpr>(first_operand_call->args[0])){
        mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(first_operand_call->args[0]);
        if (mir::isa<mir::VertexSetType>(var_expr->var.getType())) {
          while_stmt->setMetadata<bool>("swarm_frontier_convert", true);
          if (while_stmt->body->stmts->size() > 1) {
            SwitchWhileCaseFinder finder;
            finder.frontier_name = var_expr->var.getName();
            for (auto stmt: *(while_stmt->body->stmts)) {
              stmt->accept(&finder);
            }
            if (finder.can_switch) {
              while_stmt->setMetadata<bool>("swarm_switch_convert", true);
            }
          }
        }
      }
    }
  }
}

void SwarmFrontierConvert::SwitchWhileCaseFinder::visit(mir::EdgeSetApplyExpr::Ptr esae) {
  if (esae->from_func != frontier_name) {
    can_switch = false;
  }
}

void SwarmFrontierConvert::SwitchWhileCaseFinder::visit(mir::VertexSetApplyExpr::Ptr vsae) {
  if (mir::to<mir::VarExpr>(vsae->target)->var.getName() != frontier_name) {
    can_switch = false;
  }
}

void SwarmFrontierConvert::SwitchWhileCaseFinder::visit(mir::AssignStmt::Ptr assign_stmt) {
  mir::MIRVisitor::visit(assign_stmt->expr);
  if (mir::isa<mir::VarExpr>(assign_stmt->lhs)) {
    mir::VarExpr::Ptr lhs = mir::to<mir::VarExpr>(assign_stmt->lhs);
    if (mir::isa<mir::ScalarType>(lhs->var.getType())) {
      return;
    }
    can_switch = false;
  }
}

void SwarmFrontierConvert::SwitchWhileCaseFinder::visit(mir::VarDecl::Ptr var_decl) {
  mir::MIRVisitor::visit(var_decl->initVal);
  if (var_decl->name != frontier_name) {
    can_switch = false;
  }
}

}