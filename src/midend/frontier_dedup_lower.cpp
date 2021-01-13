#include <graphit/midend/frontier_dedup_lower.h>

namespace graphit {

void FrontierDedupLower::lower(void) {
  for (auto func: mir_context_->getFunctionList()) {
    ReuseFrontierFinderVisitor visitor(mir_context_);
    func->accept(&visitor);
  }
}

bool FrontierDedupLower::ReuseFrontierFinderVisitor::is_reflexive_expr(mir::AssignStmt::Ptr assign_stmt) {
  if (mir::isa<mir::VarExpr>(assign_stmt->lhs)) {
    mir::VarExpr::Ptr lhs_var = mir::to<mir::VarExpr>(assign_stmt->lhs);
    if (mir::isa<mir::VarExpr>(assign_stmt->expr)) {
      mir::VarExpr::Ptr rhs = mir::to<mir::VarExpr>(assign_stmt->expr);
      if (lhs_var->var.getName() == rhs->var.getName()) {
        return true;
      }
    }
  }
  return false;
}

void FrontierDedupLower::ReuseFrontierFinderVisitor::visit(mir::StmtBlock::Ptr stmt_block) {
  FrontierVarChangeVisitor visitor;

  std::vector<mir::Stmt::Ptr> new_stmts;
  to_deletes.clear();
  for (int i = 0; i < stmt_block->stmts->size(); i++) {
    mir::Stmt::Ptr this_stmt = (*(stmt_block->stmts))[i];

    // if this statement is an AssignStmt or a VarDecl, check to see if it uses a reusable frontier.
    if (mir::isa<mir::AssignStmt>(this_stmt)) {
      mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(this_stmt);
      if (mir::isa<mir::EdgeSetApplyExpr>(assign_stmt->expr)) {
        mir::EdgeSetApplyExpr::Ptr esae = mir::to<mir::EdgeSetApplyExpr>(assign_stmt->expr);
        if (esae->getMetadata<bool>("frontier_reusable") && esae->from_func != "") {
          visitor.frontier_name = esae->from_func;
          assign_stmt->accept(&visitor);
        }
      } else if (mir::isa<mir::VarExpr>(assign_stmt->expr)) {
        // if assignStmt lhs is the same as esae from_func (original frontier), then delete the stmt.
        mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(assign_stmt->expr);
        if (visitor.old_frontier_name != "" && var_expr->var.getName() == visitor.old_frontier_name) {
          // replace the rhs of the assign stmt with the new frontier.
          var_expr->var = mir::Var(visitor.frontier_name, var_expr->var.getType());
          if (is_reflexive_expr(assign_stmt)) {
            to_deletes.push_back(assign_stmt);
          }
        }
      }
    } else if (mir::isa<mir::VarDecl>(this_stmt)) {
      mir::VarDecl::Ptr var_decl = mir::to<mir::VarDecl>(this_stmt);
      if (var_decl->initVal != nullptr) {
        if (mir::isa<mir::EdgeSetApplyExpr>(var_decl->initVal)) {
          mir::EdgeSetApplyExpr::Ptr esae = mir::to<mir::EdgeSetApplyExpr>(var_decl->initVal);
          if (esae->getMetadata<bool>("frontier_reusable") && esae->from_func != "") {
            visitor.frontier_name = esae->from_func;
            var_decl->accept(&visitor);
          }
        }
      }
    } else if (visitor.old_frontier_name != "") {
      this_stmt->accept(&visitor);
    }
    if (std::find(to_deletes.begin(), to_deletes.end(), this_stmt) == to_deletes.end()) {
      new_stmts.push_back(this_stmt);
    }
  }
  (*(stmt_block->stmts)) = new_stmts;
  mir::MIRVisitor::visit(stmt_block);
}

void FrontierDedupLower::FrontierVarChangeVisitor::visit(mir::AssignStmt::Ptr assign_stmt) {
  mir::VarExpr::Ptr lhs_var = mir::to<mir::VarExpr>(assign_stmt->lhs);
  old_frontier_name = lhs_var->var.getName();
  lhs_var->var = mir::Var(frontier_name, lhs_var->var.getType());
}

void FrontierDedupLower::FrontierVarChangeVisitor::visit(mir::VarDecl::Ptr var_decl) {
  old_frontier_name = var_decl->name;
  var_decl->name = frontier_name;
}

void FrontierDedupLower::FrontierVarChangeVisitor::visit(mir::VertexSetApplyExpr::Ptr apply_expr) {
  mir::VarExpr::Ptr target_expr = mir::to<mir::VarExpr>(apply_expr->target);
  if (target_expr->var.getName() == old_frontier_name) {
    target_expr->var = mir::Var(frontier_name, target_expr->var.getType());
  }
}

}