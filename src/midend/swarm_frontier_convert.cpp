#include <graphit/midend/swarm_frontier_convert.h>

namespace graphit {
void SwarmFrontierConvert::analyze(void) {
  for (auto func: mir_context_->getFunctionList()) {
    func->accept(this);
  }
}

void SwarmFrontierConvert::GlobalVariableFinder::visit(mir::Call::Ptr call_ptr) {
  for (auto expr : call_ptr->args) {
    if (mir::isa<mir::VarExpr>(expr)) {
      auto var_expr = mir::to<mir::VarExpr>(expr);
      std::string var_name = var_expr->var.getName();
      if (std::find(declared_vars.begin(), declared_vars.end(), var_name) == declared_vars.end()) {
        if (var_name != frontier_name) {
          global_vars.push_back(var_expr->var);
        }
      }
    }
  }
}

void SwarmFrontierConvert::GlobalVariableFinder::visit(mir::VarDecl::Ptr var_decl) {
  if (std::find(declared_vars.begin(), declared_vars.end(), var_decl->name) == declared_vars.end()) {
      declared_vars.push_back(var_decl->name);
  }
}

void SwarmFrontierConvert::GlobalVariableFinder::visit(mir::AssignStmt::Ptr assign_stmt) {
    if (mir::isa<mir::VarExpr>(assign_stmt->lhs)) {
        mir::VarExpr::Ptr lhs = mir::to<mir::VarExpr>(assign_stmt->lhs);
        if (mir::isa<mir::ScalarType>(lhs->var.getType())) {
          if (std::find(declared_vars.begin(), declared_vars.end(), lhs->var.getName()) == declared_vars.end()) {
            if (lhs->var.getName() != frontier_name) {
              global_vars.push_back(lhs->var);
            }
          }
        }
    }
}
/*
// searches for global variables in the while body to put in the metadata so they can be passed in thru
// lambda in codegen
void attachGlobalVarToMetadata(mir::WhileStmt::Ptr while_stmt, std::string frontier_name) {
  std::vector<std::string> declared_vars;
  std::vector<mir::Var> global_vars;
  for (auto stmt: *(while_stmt->body->stmts)) {
    if (mir::isa<mir::VarDecl>(stmt)) {
      mir::VarDecl::Ptr var_decl = mir::to<mir::VarDecl>(stmt);
      if (std::find(declared_vars.begin(), declared_vars.end(), var_decl->name) == declared_vars.end()) {
        declared_vars.push_back(var_decl->name);
      }
    }
    if (mir::isa<mir::AssignStmt>(stmt)) {
      mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(stmt);
      if (mir::isa<mir::VarExpr>(assign_stmt->lhs)) {
        mir::VarExpr::Ptr lhs = mir::to<mir::VarExpr>(assign_stmt->lhs);
        if (mir::isa<mir::ScalarType>(lhs->var.getType())) {
          if (std::find(declared_vars.begin(), declared_vars.end(), lhs->var.getName()) == declared_vars.end()) {
            if (lhs->var.getName() != frontier_name) {
              global_vars.push_back(lhs->var);
            }
          }
        }
      }
    }
  }
  if (!global_vars.empty()) {
    while_stmt->setMetadata<std::vector<mir::Var>>("global_vars", global_vars);
  }
}
*/
void SwarmFrontierConvert::visit(mir::WhileStmt::Ptr while_stmt) {
  while_stmt->setMetadata<bool>("swarm_frontier_convert", false);
  while_stmt->setMetadata<bool>("swarm_switch_convert", false);

  if (mir::isa<mir::EqExpr>(while_stmt->cond)) {
    mir::EqExpr::Ptr cond_expr = mir::to<mir::EqExpr>(while_stmt->cond);
    if (mir::isa<mir::Call>(cond_expr->operands[0])) {
      mir::Call::Ptr first_operand_call = mir::to<mir::Call>(cond_expr->operands[0]);
      if (mir::isa<mir::VarExpr>(first_operand_call->args[0])){
        mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(first_operand_call->args[0]);
        if (mir::isa<mir::VertexSetType>(var_expr->var.getType()) || mir::isa<mir::PriorityQueueType>(var_expr->var.getType())) {
          // if the while stmt condition is tracking if a frontier is empty or not, then we are in a convertible
          // while loop.
          while_stmt->setMetadata<bool>("swarm_frontier_convert", true);
	  if (mir::isa<mir::PriorityQueueType>(var_expr->var.getType())) {
            while_stmt->setMetadata<bool>("update_priority_queue", true);
	  }
	  while_stmt->setMetadata<mir::Var>("swarm_frontier_var", var_expr->var); 
	  //attachGlobalVarToMetadata(while_stmt, var_expr->var.getName());
	  GlobalVariableFinder global_variable_finder;
	  global_variable_finder.current_while_stmt = while_stmt;
	  global_variable_finder.frontier_name = var_expr->var.getName();

 	  while_stmt->body->accept(&global_variable_finder);
    	  while_stmt->setMetadata<std::vector<mir::Var>>("global_vars", global_variable_finder.global_vars);
          
	  RoundParamEmitter round_param_emitter;
          round_param_emitter.current_while_stmt = while_stmt;
          while_stmt->body->accept(&round_param_emitter);
          round_param_emitter.fill_update_size_stmts();

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

              // Figure out which statements are vertex vs frontier level.
              SwarmSwitchCaseSeparator separator;
              separator.current_while_stmt = while_stmt;
              for (auto stmt: *(while_stmt->body->stmts)) {
                stmt->accept(&separator);
                separator.idx++;
              }
              separator.fill_frontier_stmts();

              // Converts statements to switch statements and stores them in either the while body
              // or in while stmt metadata
              separator.setup_switch_cases();

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

void SwarmFrontierConvert::RoundParamEmitter::visit(mir::StmtBlock::Ptr stmt_block) {
	int i = 0;
	for (auto stmt: *(stmt_block->stmts)) {
		stmt->accept(this);
		if (insert_found) {
			insert_stmt_idxs.push_back(i);
		}
		insert_found = false;
		i++;
	}

}
void SwarmFrontierConvert::RoundParamEmitter::visit(mir::Call::Ptr call_ptr) {
  if (call_ptr->name == "builtin_insert") {
    std::vector<mir::Var> round_vars;
    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      round_vars = current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars");
    }
    std::vector<mir::Expr::Ptr> init_expr_stmts;
    if (current_while_stmt->hasMetadata<std::vector<mir::Expr::Ptr>>("init_expr_stmts")) {
      init_expr_stmts = current_while_stmt->getMetadata<std::vector<mir::Expr::Ptr>>("init_expr_stmts");
    }

    // Push an additional variable indicating the VFL round, which will be incremented on, to the whilestmt metadata.
    auto int_type = std::make_shared<mir::ScalarType>();
    int_type->type = mir::ScalarType::Type::INT;
    mir::Var new_var = mir::Var("insert_round_" + std::to_string(curr_no), int_type);
    call_ptr->setMetadata<mir::Var>("increment_round_var", new_var);

    round_vars.push_back(new_var);
    current_while_stmt->setMetadata<std::vector<mir::Var>>("add_src_vars", round_vars);
    
    // Queue the initial state of the above variable to metadata, which should be VFL size, so codegen
    // can correctly queue initial vertices as {src, VFL.get_size()}
    mir::Call::Ptr get_size_call = std::make_shared<mir::Call>();
    get_size_call->name = "builtin_get_size";
    get_size_call->args.push_back(call_ptr->args[0]);
    
    init_expr_stmts.push_back(get_size_call);
    current_while_stmt->setMetadata<std::vector<mir::Expr::Ptr>>("init_expr_stmts", init_expr_stmts);
    curr_no++;

    // Update state to add VFL.update_size calls later in second pass.
    insert_stmt_frontier_list_vars.push_back(call_ptr->args[0]);
    insert_stmt_incr_vars.push_back(new_var);
    insert_found = true;
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

void SwarmFrontierConvert::SwarmSwitchCaseSeparator::setup_switch_cases() {
  // If vertex / frontier statements need to be separated, then store new stmtblocks in the metadata
  // and build up each by iterating through each statement, inserting blank statements when necessary.
  mir::StmtBlock::Ptr new_single_level = std::make_shared<mir::StmtBlock>();
  mir::StmtBlock::Ptr new_frontier_level = std::make_shared<mir::StmtBlock>();

  if (is_bucket_queue()) {
    int s_ptr = 0;  // ptr in single vertex statements
    int f_ptr = 0;  // ptr in frontier level statements
    int round_num = 0;

    auto single_level = swarm_single_level;
    auto frontier_level = swarm_frontier_level;

    while (s_ptr < single_level.size() || f_ptr < frontier_level.size()) {
      // if vertex level is ahead of frontier level, then insert a blank into the vertex level block.
      if (s_ptr >= single_level.size() || (single_level[s_ptr] > frontier_level[f_ptr] && f_ptr < frontier_level.size())) {
        new_frontier_level->insertStmtEnd(convert_to_switch_stmt(frontier_level[f_ptr], round_num, false));
        new_single_level->insertStmtEnd(create_blank_switch_stmt(round_num, true));
        f_ptr++;
       // if frontier level is ahead of vertex level, then check to see whether frontier task immediately
       // follows vertex level (which is ok). Otherwise, push a blank into frontier level block.
      } else if (f_ptr >= frontier_level.size() || single_level[s_ptr] < frontier_level[f_ptr]) {
        if (f_ptr >= frontier_level.size() || frontier_level[f_ptr] - single_level[s_ptr] > 1) {
          new_frontier_level->insertStmtEnd(create_blank_switch_stmt(round_num, false));
          new_single_level->insertStmtEnd(convert_to_switch_stmt(single_level[s_ptr], round_num, true));
        } else {
          new_frontier_level->insertStmtEnd(convert_to_switch_stmt(frontier_level[f_ptr], round_num, false));
          new_single_level->insertStmtEnd(convert_to_switch_stmt(single_level[s_ptr], round_num, true));
          f_ptr++;
        }
        s_ptr++;
      }
      round_num++;
    }
  } else {
    // if no statements are frontier level, then simply modify while stmt block in-place.
    //std::vector<mir::Stmt::Ptr> new_stmts;
    for (int i = 0; i < current_while_stmt->body->stmts->size(); i++) {
      //new_stmts.push_back(convert_to_switch_stmt(i, i, true));
      new_single_level->insertStmtEnd(convert_to_switch_stmt(i, i, true));
    }
    //(*(current_while_stmt->body->stmts)) = new_stmts;
    new_frontier_level->stmts = new std::vector<mir::Stmt::Ptr>();
  }
  current_while_stmt->setMetadata<mir::StmtBlock::Ptr>("new_single_bucket", new_single_level);
  current_while_stmt->setMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket", new_frontier_level);
}

}
