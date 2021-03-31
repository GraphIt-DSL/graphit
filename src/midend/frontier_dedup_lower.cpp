#include <graphit/midend/frontier_dedup_lower.h>

namespace graphit {

void FrontierDedupLower::lower(void) {
  for (auto func: mir_context_->getFunctionList()) {
    ReuseFrontierFinderVisitor visitor(mir_context_);
    VertexDeduplicationVisitor vertex_deduplication_visitor(mir_context_);
    func->accept(&visitor);
    func->accept(&vertex_deduplication_visitor);
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

void FrontierDedupLower::DedupVectorAttacher::visit(mir::EnqueueVertex::Ptr enqueue_vertex) {
	enqueue_vertex->setMetadata<mir::Var>("dedup_vector", dedup_vector_var);
}

void FrontierDedupLower::VertexDeduplicationVisitor::visit(mir::StmtBlock::Ptr stmt_block) {
  std::vector<mir::Stmt::Ptr> new_stmts;
  for (int i = 0; i < stmt_block->stmts->size(); i++) {
    mir::Stmt::Ptr this_stmt = (*(stmt_block->stmts))[i];
    new_stmts.push_back(this_stmt);
    // if this statement is an AssignStmt or a VarDecl, check to see if it uses a reusable frontier.
    if (mir::isa<mir::VarDecl>(this_stmt)) {
      mir::VarDecl::Ptr var_decl = mir::to<mir::VarDecl>(this_stmt);
      if (mir::isa<mir::EdgeSetApplyExpr>(var_decl->initVal)) {
        mir::EdgeSetApplyExpr::Ptr esae = mir::to<mir::EdgeSetApplyExpr>(var_decl->initVal);
        if (esae->getMetadata<bool>("enable_deduplication") && esae->from_func != "") {
          frontier_name = esae->from_func;
          auto inFrontierType = std::make_shared<mir::ScalarType>();
          std::string array_name = "in_frontier_" + mir_context_->getUniqueNameCounterString();
	  auto bool_type = std::make_shared<mir::ScalarType>();
	  bool_type->type = mir::ScalarType::Type::BOOL;
	  mir::Var vector_var = mir::Var(array_name, bool_type);
          esae->setMetadata<mir::Var>("dedup_vector", vector_var);

          std::string new_reset_frontier_fxn_name = "reset_" + array_name;
          inFrontierType->type = mir::ScalarType::Type ::BOOL;
          mir::VertexSetApplyExpr::Ptr dedup_vsae = std::make_shared<mir::VertexSetApplyExpr>(frontier_name, inFrontierType, new_reset_frontier_fxn_name);
          dedup_vsae->setMetadata<bool>("requires_output", false);
          dedup_vsae->setMetadata<bool>("inline_function", true);
          auto new_expr_stmt = std::make_shared<mir::ExprStmt>();
          new_expr_stmt->expr = dedup_vsae;

          auto new_func_decl = std::make_shared<mir::FuncDecl>();
          new_func_decl->name = new_reset_frontier_fxn_name;
          mir::ElementType::Ptr v_type = std::make_shared<mir::ElementType>();
          mir::VectorType::Ptr vector_type = std::make_shared<mir::VectorType>();
          v_type->ident = "Vertex";
          new_func_decl->args.push_back(mir::Var("v", v_type));
          vector_type->element_type = v_type->clone<mir::ElementType>();

          auto assign_stmt = std::make_shared<mir::AssignStmt>();
          assign_stmt->lhs = std::make_shared<mir::TensorReadExpr>(array_name,
              "v",vector_type,
              v_type->clone<mir::ElementType>());
          assign_stmt->expr = std::make_shared<mir::BoolLiteral>();
          new_func_decl->body = std::make_shared<mir::StmtBlock>();
          new_func_decl->body->insertStmtEnd(assign_stmt);
          new_func_decl->setMetadata<bool>("inline_only", true);  // don't declare a global version in codegen
          mir_context_->addFunction(new_func_decl);
          new_stmts.push_back(new_expr_stmt);

	  DedupVectorAttacher dedup_vector_attacher;
	  dedup_vector_attacher.dedup_vector_var = vector_var;
	  mir_context_->functions_map_[esae->input_function_name]->accept(&dedup_vector_attacher);
        }
      }
    }
  }
  (*(stmt_block->stmts)) = new_stmts;
  mir::MIRVisitor::visit(stmt_block);
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
    } else if (mir::isa<mir::ExprStmt>(this_stmt)) {
      mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(stmt_block->stmts))[i]);
      if (mir::isa<mir::Call>(expr_stmt->expr)) {
        mir::Call::Ptr call_expr = mir::to<mir::Call>(expr_stmt->expr);
        if (call_expr->name == "deleteObject" && mir::isa<mir::VarExpr>(call_expr->args[0]) && mir::to<mir::VarExpr>(call_expr->args[0])->var.getName() == visitor.frontier_name) {
          to_deletes.push_back(expr_stmt);
          continue;
        }
      }
      this_stmt->accept(&visitor);
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
  if (mir::isa<mir::VarExpr>(assign_stmt->lhs)) {
    mir::VarExpr::Ptr lhs_var = mir::to<mir::VarExpr>(assign_stmt->lhs);
    old_frontier_name = lhs_var->var.getName();
    lhs_var->var = mir::Var(frontier_name, lhs_var->var.getType());
  }
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

void FrontierDedupLower::FrontierVarChangeVisitor::visit(mir::Call::Ptr call_expr) {
  for (auto arg : call_expr->args) {
    if (mir::isa<mir::VarExpr>(arg)) {
      mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(arg);
      if (var_expr->var.getName() == old_frontier_name) {
        var_expr->var = mir::Var(frontier_name, var_expr->var.getType());
      }
    }
  }
}

}
