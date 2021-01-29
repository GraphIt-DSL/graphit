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
          // if the while stmt condition is tracking if a frontier is empty or not, then we are in a convertible
          // while loop.
          while_stmt->setMetadata<bool>("swarm_frontier_convert", true);
          while_stmt->setMetadata<mir::Var>("swarm_frontier_var", var_expr->var);
          if (while_stmt->body->stmts->size() > 1) {
            // First, determine whether the while loop can be converted to switch cases.
            SwitchWhileCaseFinder finder;
            finder.frontier_name = var_expr->var.getName();
            for (auto stmt: *(while_stmt->body->stmts)) {
              stmt->accept(&finder);
            }
            if (finder.can_switch) {
              // If it is, then we want to figure out how to push statements to the swarm queue
              // by separating vertex from frontier level operators/exprs.
              while_stmt->setMetadata<bool>("swarm_switch_convert", true);

              RuntimeInsertConvert runtime_convert;
              while_stmt->body->accept(&runtime_convert);
              while_stmt->body = runtime_convert.new_stmt_block;

              SwarmSwitchCaseSeparator separator;
              separator.current_while_stmt = while_stmt;
              for (auto stmt: *(while_stmt->body->stmts)) {
                stmt->accept(&separator);
                stmt->setMetadata<int>("while_stmt_idx", separator.idx);
                separator.idx++;
              }
              separator.fill_frontier_stmts();

              // Then wrap each stmt in a SwarmSwitchStmt.
              int round = 0;
              std::vector<mir::Stmt::Ptr> new_stmts;
              for (int i = 0; i < while_stmt->body->stmts->size(); i++) {
                mir::Stmt::Ptr stmt = (*(while_stmt->body->stmts))[i];
                mir::SwarmSwitchStmt::Ptr switch_stmt = std::make_shared<mir::SwarmSwitchStmt>();
                switch_stmt->round = round;
                switch_stmt->stmt = stmt;
                new_stmts.push_back(switch_stmt);
                round++;
              }
              (*(while_stmt->body->stmts)) = new_stmts;
              mir::MIRVisitor::visit(while_stmt->body);
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

void SwarmFrontierConvert::RuntimeInsertConvert::visit(mir::StmtBlock::Ptr stmt_block) {
  for (auto stmt : *stmt_block->stmts) {
    stmt->accept(this);
    if (mir::isa<mir::ExprStmt>(stmt)) {
      mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>(stmt);
      if (!mir::isa<mir::Call>(expr_stmt->expr)) {
        new_stmt_block->insertStmtEnd(stmt);
      }
    } else {
      new_stmt_block->insertStmtEnd(stmt);
    }
  }
}

void SwarmFrontierConvert::RuntimeInsertConvert::visit(mir::Call::Ptr call_ptr) {
  mir::ExprStmt::Ptr call_stmt = std::make_shared<mir::ExprStmt>();
  call_stmt->expr = call_ptr;
  new_stmt_block->insertStmtEnd(call_stmt);
  if (call_ptr->name == "builtin_insert") {
    mir::Call::Ptr new_round_inc_call = std::make_shared<mir::Call>();
    new_round_inc_call->name = "builtin_increment_round";
    new_round_inc_call->args.push_back(call_ptr->args[0]);
    mir::ExprStmt::Ptr stmt1 = std::make_shared<mir::ExprStmt>();
    stmt1->expr = new_round_inc_call;
    new_stmt_block->insertStmtEnd(stmt1);
  }
}

void SwarmFrontierConvert::SwarmSwitchCaseSeparator::visit(mir::VertexSetApplyExpr::Ptr vsae) {
  if (mir::to<mir::VarExpr>(vsae->target)->var.getName() == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName()) {
    insert_single_source_case(idx);
  }
}

void SwarmFrontierConvert::SwarmSwitchCaseSeparator::visit(mir::AssignStmt::Ptr assign_stmt) {
  if (mir::isa<mir::VarExpr>(assign_stmt->lhs)) {
    mir::VarExpr::Ptr lhs = mir::to<mir::VarExpr>(assign_stmt->lhs);
    if (lhs->var.getName() == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName()) {
      insert_single_source_case(idx);
    }
  }
}

void SwarmFrontierConvert::SwarmSwitchCaseSeparator::visit(mir::VarDecl::Ptr var_decl) {
  if (var_decl->name == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName()) {
    insert_single_source_case(idx);
  }
}

void SwarmFrontierConvert::SwarmSwitchCaseSeparator::visit(mir::Call::Ptr call_expr) {
  for (const auto arg : call_expr->args) {
    if (mir::isa<mir::VarExpr>(arg)) {
      mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(arg);
      if (var_expr->var.getName() == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName()) {
        insert_single_source_case(idx);
      }
    }
  }
}

}