#include <graphit/backend/codegen_swarm/codegen_swarm.h>
#include <graphit/midend/mir.h>


namespace graphit {
int CodeGenSwarm::genSwarmCode(void) {
  genIncludeStmts();
  oss << "int __argc;" << std::endl;
  oss << "char **__argv;" << std::endl;

  genEdgeSets();
  genConstants();
  std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

  for (auto it = functions.begin(); it != functions.end(); it++) {
    it->get()->accept(frontier_finder);
  }

//  genSwarmFrontiers();

  for (auto it = functions.begin(); it != functions.end(); it++) {
    it->get()->accept(this);
  }

  genMainFunction();
  return 0;
}

int CodeGenSwarm::genIncludeStmts(void) {
  oss << "#include <tuple>" << std::endl;
  oss << "#include \"swarm_intrinsics.h\"" << std::endl;
  oss << "#include \"scc/queues.h\"" << std::endl;
  oss << "#include \"scc/autoparallel.h\"" << std::endl;
}

int CodeGenSwarm::genEdgeSets(void) {
  for (auto edgeset : mir_context_->getEdgeSets()) {
    auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
    edge_set_type->accept(this);
    oss << " " << edgeset->name << ";" << std::endl;
  }
}

//int CodeGenSwarm::genSwarmFrontiers() {
//  for (std::string frontier_name : frontier_finder->swarm_frontier_prioq_vars) {
//    auto frontier_var = frontier_finder->edgeset_var_map[frontier_name];
//    auto vertex_set_type = mir::to<mir::VertexSetType>(frontier_var->type);
//    oss << "swarm::PrioQueue<";
//    oss << "int32_t";
//    oss << "> swarm_" << frontier_name << ";" << std::endl;
//  }
//  for (std::string frontier_name : frontier_finder->swarm_frontier_bucketq_vars) {
//    auto frontier_var = frontier_finder->edgeset_var_map[frontier_name];
//    auto vertex_set_type = mir::to<mir::VertexSetType>(frontier_var->type);
//    oss << "swarm::BucketQueue<";
//    oss << "int32_t";
//    oss << "> swarm_" << frontier_name << ";" << std::endl;
//  }
//}

void CodeGenSwarmFrontierFinder::visit(mir::VarDecl::Ptr var_decl) {
  if (mir::isa<mir::VertexSetType>(var_decl->type)) {
    std::string name = var_decl->name;\
    edgeset_var_map[name] = var_decl;
  }
}

void CodeGenSwarmFrontierFinder::visit(mir::WhileStmt::Ptr while_stmt) {
  if (while_stmt->getMetadata<bool>("swarm_frontier_convert")) {
    auto frontier_var = while_stmt->getMetadata<mir::Var>("swarm_frontier_var");

    // if converting to switch statements, then you might need to use the BucketQueue.
    if (while_stmt->getMetadata<bool>("swarm_switch_convert")
        && while_stmt->hasMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")) {
      // why is this here
      if (edgeset_var_map.find(frontier_var.getName()) != edgeset_var_map.end()) {
        swarm_frontier_bucketq_vars.push_back(frontier_var.getName());
      }
    } else {
      // why is this here
      if (edgeset_var_map.find(frontier_var.getName()) != edgeset_var_map.end()) {
        swarm_frontier_prioq_vars.push_back(frontier_var.getName());
      }
    }
    if (edgeset_var_map.find(frontier_var.getName()) != edgeset_var_map.end()) {
      if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
        auto add_src_vars = while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars");
        edgeset_var_map[frontier_var.getName()]->setMetadata<std::vector<mir::Var>>("add_src_vars", add_src_vars);
      }
      edgeset_var_map[frontier_var.getName()]->setMetadata<mir::Var>("swarm_frontier_var", while_stmt->getMetadata<mir::Var>("swarm_frontier_var"));
    }
  }
}

void CodeGenSwarm::visit(mir::EdgeSetType::Ptr edge_set_type) {
  if (edge_set_type->weight_type == nullptr) {
    oss << "swarm_runtime::GraphT<int>";
  } else {
    oss << "swarm_runtime::GraphT<";
    edge_set_type->weight_type->accept(this);
    oss << ">";
  }
}

void CodeGenSwarm::visit(mir::ScalarType::Ptr scalar_type) {
  switch (scalar_type->type) {
    case mir::ScalarType::Type::INT:
      oss << "int";
      break;
    case mir::ScalarType::Type::UINT:
      oss << "unsigned int";
      break;
    case mir::ScalarType::Type::UINT_64:
      oss << "uint64_t";
      break;
    case mir::ScalarType::Type::FLOAT:
      oss << "float";
      break;
    case mir::ScalarType::Type::DOUBLE:
      oss << "double";
      break;
    case mir::ScalarType::Type::BOOL:
      oss << "bool";
      break;
    case mir::ScalarType::Type::STRING:
      oss << "string";
      break;
    default:
      break;
  }
}


int CodeGenSwarm::genConstants(void) {
  for (auto constant : mir_context_->getLoweredConstants()) {
    if (mir::isa<mir::VectorType>(constant->type)) {
      mir::VectorType::Ptr vector_type = mir::to<mir::VectorType>(constant->type);
      vector_type->vector_element_type->accept(this);
      oss << " *" << constant->name << ";" << std::endl;
    } else if (mir::isa<mir::ScalarType>(constant->type)) {
      constant->type->accept(this);
      oss << " " << constant->name;
      oss << ";" << std::endl;
    } else if (mir::isa<mir::PriorityQueueType>(constant->type)) {
      oss << "swarm::BucketQueue<int> swarm_" << constant->name << ";" << std::endl;
      oss << "int swarm_" << constant->name << "_delta;" << std::endl;
    }
  }
}

void CodeGenSwarm::visit(mir::IntLiteral::Ptr literal) {
  oss << literal->val;
}
void CodeGenSwarm::visit(mir::FloatLiteral::Ptr literal) {
  oss << literal->val;
}
void CodeGenSwarm::visit(mir::BoolLiteral::Ptr literal) {
  oss << "(bool)" << literal->val;
}
void CodeGenSwarm::visit(mir::StringLiteral::Ptr literal) {
  oss << "\"" << literal->val << "\"";
}

void CodeGenSwarm::visitBinaryExpr(mir::BinaryExpr::Ptr expr, std::string op) {
  oss << "(";
  expr->lhs->accept(this);
  oss << " " << op << " ";
  expr->rhs->accept(this);
  oss << ")";
}
void CodeGenSwarm::visit(mir::AndExpr::Ptr expr) {
  visitBinaryExpr(expr, "&&");
}
void CodeGenSwarm::visit(mir::OrExpr::Ptr expr) {
  visitBinaryExpr(expr, "||");
}
void CodeGenSwarm::visit(mir::XorExpr::Ptr expr) {
  visitBinaryExpr(expr, "^");
}
void CodeGenSwarm::visit(mir::NotExpr::Ptr expr) {
  oss << "!(";
  expr->operand->accept(this);
  oss << ")";
}
void CodeGenSwarm::visit(mir::MulExpr::Ptr expr) {
  visitBinaryExpr(expr, "*");
}
void CodeGenSwarm::visit(mir::DivExpr::Ptr expr) {
  visitBinaryExpr(expr, "/");
}
void CodeGenSwarm::visit(mir::AddExpr::Ptr expr) {
  visitBinaryExpr(expr, "+");
}
void CodeGenSwarm::visit(mir::SubExpr::Ptr expr) {
  visitBinaryExpr(expr, "-");
}
void CodeGenSwarm::visit(mir::NegExpr::Ptr expr) {
  if (expr->negate)
    oss << "-";
  oss << "(";
  expr->operand->accept(this);
  oss << ")";
}
void CodeGenSwarm::visit(mir::VarExpr::Ptr expr) {
  if (expr->var.getName() == "argc")
    oss << "__argc";
  else if (expr->var.getName() == "argv")
    oss << "__argv";
  else
    oss << expr->var.getName();
}
void CodeGenSwarm::visit(mir::Call::Ptr call) {
  if (call->name.find("builtin_") == 0 || call->name == "startTimer" || call->name == "stopTimer")
    oss << "swarm_runtime::";
  oss << call->name << "(";
  bool printDelimeter = false;
  for (auto expr: call->args) {
    if (printDelimeter)
      oss << ", ";
    expr->accept(this);

//    // lowkey hack to add an extra parameter into the runtime lib call to add a vertex into the swarm frontier too.
//    if (call->name == "builtin_addVertex" && mir::isa<mir::VarExpr>(expr)) {
//      auto var_expr = mir::to<mir::VarExpr>(expr);
//      if (frontier_finder->isa_frontier(var_expr->var.getName())) {
//        oss << ", swarm_" << var_expr->var.getName();
//      }
//    }
    printDelimeter = true;
  }
  oss << ")";
}

int CodeGenSwarm::genMainFunction(void) {
  oss << "int main(int argc, char* argv[]) {" << std::endl;
  indent();
  printIndent();
  oss << "__argc = argc;" << std::endl;
  printIndent();
  oss << "__argv = argv;" << std::endl;

  for (auto stmt : mir_context_->edgeset_alloc_stmts) {
    mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(stmt);
    mir::EdgeSetLoadExpr::Ptr edge_set_load_expr = mir::to<mir::EdgeSetLoadExpr>(assign_stmt->expr);
    mir::VarExpr::Ptr lhs_var = mir::to<mir::VarExpr>(assign_stmt->lhs);
    std::string var_name = lhs_var->var.getName();
    printIndent();
    oss << "swarm_runtime::load_graph(" << var_name << ", ";
    edge_set_load_expr->file_name->accept(this);
    oss << ");" << std::endl;
  }
  for (auto constant : mir_context_->getLoweredConstants()) {
    if (mir::isa<mir::ScalarType>(constant->type) && constant->initVal != nullptr) {
      printIndent();
      oss << constant->name;
      oss << " = ";
      constant->initVal->accept(this);
      oss << ";" << std::endl;
    } else if (mir::isa<mir::VectorType>(constant->type)) {
      mir::VectorType::Ptr vector_type = mir::to<mir::VectorType>(constant->type);
      printIndent();
      oss << constant->name;
      oss << " = new ";
      vector_type->vector_element_type->accept(this);
      oss << "[";
      mir_context_->getElementCount(vector_type->element_type)->accept(this);
      oss << "];" << std::endl;
    }

  }
  for (auto stmt : mir_context_->field_vector_init_stmts) {
    stmt->accept(this);
  }
  printIndent();
//  oss << "pls::enqueue(swarm_main, 0, NOHINT);" << std::endl;
  printIndent();
//  oss << "pls::run();" << std::endl;
  oss << "SCC_PARALLEL( swarm_main(); );" << std::endl;
  dedent();
  oss << "}" << std::endl;

}
void CodeGenSwarm::visit(mir::ExprStmt::Ptr stmt) {
  printIndent();
  stmt->expr->accept(this);
  oss << ";" <<std::endl;
}

void CodeGenSwarm::visit(mir::EdgeSetLoadExpr::Ptr expr) {
  assert(false && "EdgeSetLoadExpr should directly be handled in main");
}

void CodeGenSwarm::visit(mir::VertexSetType::Ptr vertexset_type) {
  oss << "swarm_runtime::VertexFrontier";
}

void CodeGenSwarm::visit(mir::ListType::Ptr list_type) {
  if (mir::isa<mir::VertexSetType>(list_type->element_type)) {
    oss << "swarm_runtime::VertexFrontierList";
    return;
  }
  oss << "std::vector<";
  list_type->element_type->accept(this);
  oss << ">";
}

void CodeGenSwarm::visit(mir::ListAllocExpr::Ptr alloc_expr) {
  if (mir::isa<mir::VertexSetType>(alloc_expr->element_type)) {
    oss << "swarm_runtime::create_new_vertex_frontier_list(";
    mir::VertexSetType::Ptr vst = mir::to<mir::VertexSetType>(alloc_expr->element_type);
    mir::Expr::Ptr size_expr = mir_context_->getElementCount(vst->element);
    size_expr->accept(this);
    oss << ")";
    return;
  }
  oss << "std::vector<";
  alloc_expr->element_type->accept(this);
  oss << ">()";
}

void CodeGenSwarm::visit_assign_stmt(mir::AssignStmt::Ptr stmt) {
  if (mir::isa<mir::PriorityQueueAllocExpr>(stmt->expr)) {
    mir::PriorityQueueAllocExpr::Ptr pqae = mir::to<mir::PriorityQueueAllocExpr>(stmt->expr);
    auto associated_element_type_size = mir_context_->getElementCount(pqae->element_type);
    printIndent();
    oss << "swarm_";
    stmt->lhs->accept(this);
    oss << "_delta = ";
    if (pqae->getMetadata<int>("delta") > 0) {
      oss << pqae->getMetadata<int>("delta");
    } else {
      oss << "atoi(__argv[" << -pqae->getMetadata<int>("delta") << "])";
    }
    oss << ";" << std::endl;
    printIndent();
    oss << "swarm_";
    stmt->lhs->accept(this);
    oss << ".push_init(0, ";
    pqae->starting_node->accept(this);
    oss << ");" << std::endl;

    return;
  }
  printIndent();
  stmt->lhs->accept(this);
  oss << " = ";
  stmt->expr->accept(this);
  oss << ";" << std::endl;
}

void CodeGenSwarm::visit(mir::AssignStmt::Ptr stmt) {
	visit_assign_stmt(stmt);
}
void CodeGenSwarm::visit(mir::TensorArrayReadExpr::Ptr expr) {
  expr->target->accept(this);
  oss << "[";
  expr->index->accept(this);
  oss << "]";
}
void CodeGenSwarm::visit(mir::FuncDecl::Ptr func) {
  if (func->name == "main")
    oss << "SWARM_FUNC_ATTRIBUTES" << std::endl;
  if (func->result.getType())
    func->result.getType()->accept(this);
  else
    oss << "void";
  oss << " ";
  if (func->name == "main")
    oss << "swarm_main";
  else
    oss << func->name;

  oss << "(";
  if (func->name != "main") {
    bool printDelim = false;
    for (auto arg: func->args) {
      if (printDelim)
        oss << ", ";
      arg.getType()->accept(this);
      oss << " ";
      oss << arg.getName();
      printDelim = true;
    }
  } else {
//    oss << "pls::Timestamp";
  }
  oss << ") {" << std::endl;
  indent();

  // copied from gpu
  if (func->result.isInitialized()) {
    printIndent();
    func->result.getType()->accept(this);
    oss << " " << func->result.getName() << ";" << std::endl;
  }
  // end copied from gpu

  func->body->accept(this);

  // copied from gpu
  if (func->result.isInitialized()) {
    printIndent();
    oss << "return " << func->result.getName() << ";" << std::endl;
  }
  // end copied from gpu

  dedent();
  oss << "}" << std::endl;
}

void CodeGenSwarm::visit(mir::ElementType::Ptr elem) {
  oss << "int";
}

void CodeGenSwarm::visit(mir::StmtBlock::Ptr stmts) {
  for (auto stmt: *(stmts->stmts))
    stmt->accept(this);
}
void CodeGenSwarm::visit(mir::IfStmt::Ptr if_stmt) {
	printIndent();
	oss << "if (";
	if_stmt->cond->accept(this);
	oss << ") {" << std::endl;
	indent();
	if_stmt->ifBody->accept(this);
	dedent();
	printIndent();
	oss << "}";
	if (if_stmt->elseBody != nullptr) {
		oss << " else {" << std::endl;
		indent();
		if_stmt->elseBody->accept(this);
		dedent();
		printIndent();
		oss << "}";
	}	
	oss << std::endl;
}
void CodeGenSwarm::visit(mir::PrintStmt::Ptr stmt) {
  printIndent();
  oss << "swarm_runtime::print(";
  stmt->expr->accept(this);
  oss << ");" << std::endl;
}

void CodeGenSwarm::visit(mir::EqExpr::Ptr eq_expr) {
  oss << "(";
  eq_expr->operands[0]->accept(this);
  oss << ")";

  for (unsigned i = 0; i < eq_expr->ops.size(); ++i) {
    switch(eq_expr->ops[i]) {
      case mir::EqExpr::Op::LT:
        oss << " < ";
        break;
      case mir::EqExpr::Op::LE:
        oss << " <= ";
        break;
      case mir::EqExpr::Op::GT:
        oss << " > ";
        break;
      case mir::EqExpr::Op::GE:
        oss << " >= ";
        break;
      case mir::EqExpr::Op::EQ:
        oss << " == ";
        break;
      case mir::EqExpr::Op::NE:
        oss << " != ";
        break;
      default:
        assert(false && "Invalid operator for EqExpr\n");

    }
    oss << "(";
    eq_expr->operands[i+1]->accept(this);
    oss << ")";
  }
}

void CodeGenSwarm::visit(mir::VertexSetAllocExpr::Ptr vsae) {
  mir::Expr::Ptr size_expr = mir_context_->getElementCount(vsae->element_type);
  oss << "swarm_runtime::create_new_vertex_set(";
  size_expr->accept(this);
  oss << ", ";
  if (vsae->size_expr == nullptr)
    oss << "0";
  else
    vsae->size_expr->accept(this);
  oss << ")";
}

void CodeGenSwarm::visit(mir::VarDecl::Ptr stmt) {
  // don't need to reinitialize the frontiers declaration if already declared outside in swarm
  // If a frontier is declared, then just declare the array version of the frontier.
  // Edgeset operations should be applied on the Swarm frontier, so no need to generate frontier array.
  if (frontier_finder->isa_frontier(stmt->name)) {
    // Just generate the Edgeset apply operator code
    if (mir::isa<mir::EdgeSetApplyExpr>(stmt->initVal)) {
      stmt->initVal->accept(this);
      return;
    }
    printIndent();
    oss << "swarm::BucketQueue<";
    // The vertex type (usually) for the frontier
    if (stmt->hasMetadata<mir::Var>("frontier_var")) {
      stmt->getMetadata<mir::Var>("frontier_var").getType()->accept(this);
    } else {
      oss << "int";
    }

    // The extra types that are passed task to task
    if (stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      for (auto add_src_var : stmt->getMetadata<std::vector<mir::Var>>("add_src_vars")) {
        oss << ", ";
	add_src_var.getType()->accept(this);
      }
    }    
    oss << "> swarm_" << stmt->name << ";" << std::endl;
  }
  printIndent();
  stmt->type->accept(this);
  oss << " " << stmt->name;
  if (stmt->initVal != nullptr) {
    oss << " = ";
    stmt->initVal->accept(this);
  }
  oss << ";" << std::endl;

}
void CodeGenSwarmQueueEmitter::visit(mir::VarDecl::Ptr stmt) {
  if (stmt->name == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName()) {
    stmt->initVal->accept(this);
    return;
  }
  printIndent();
  stmt->type->accept(this);
  oss << " " << stmt->name;
  if (stmt->initVal != nullptr) {
    oss << " = ";
    stmt->initVal->accept(this);
  }
  oss << ";" << std::endl;
}

void CodeGenSwarmQueueEmitter::visit(mir::AssignStmt::Ptr stmt) {
  if (mir::isa<mir::VarExpr>(stmt->lhs)) {
    auto lhs_name = mir::to<mir::VarExpr>(stmt->lhs)->var.getName();
    if (lhs_name == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName()) {
      stmt->expr->accept(this);
      return;
    }
  }
  visit_assign_stmt(stmt);
}

void CodeGenSwarmQueueEmitter::visit(mir::SwarmSwitchStmt::Ptr switch_stmt) {
  printIndent();
  oss << "case " << std::to_string(switch_stmt->round) << ": {" << std::endl;
  indent();
  switch_stmt->stmt_block->accept(this);
  if (switch_stmt->getMetadata<bool>("is_vertex_level")) {
    if (!push_inserted) {
      printPushStatement(false, true);
      is_insert_call = false;
    }
  }

  printIndent();
  oss << "break;" << std::endl;
  dedent();
  printIndent();
  oss << "}" << std::endl;
}

void CodeGenSwarmQueueEmitter::visit(mir::Call::Ptr call) {
  if (call->name == "builtin_insert") {
    if (call->hasMetadata<mir::Var>("increment_round_var")) {
      is_insert_call = true;
    }
  }
  if (call->name.find("builtin_") == 0 || call->name == "startTimer" || call->name == "stopTimer")
    oss << "swarm_runtime::";
  oss << call->name << "(";
  bool printDelimeter = false;
  for (auto expr: call->args) {
    if (printDelimeter)
      oss << ", ";
    expr->accept(this);
    printDelimeter = true;
  }
  if (call->hasMetadata<mir::Var>("increment_round_var")) {
    oss << ", ";
    printRoundIncrement();
  }
  oss << ")";
}


void CodeGenSwarmQueueEmitter::visit(mir::VarExpr::Ptr expr) {
  if (expr->var.getName() == "argc")
    oss << "__argc";
  else if (expr->var.getName() == "argv")
    oss << "__argv";
  else if (expr->var.getName() == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName())
    printSrcVertex();
  else
    oss << expr->var.getName();
}

void CodeGenSwarmQueueEmitter::visit(mir::VertexSetType::Ptr vertexset_type) {
  vertexset_type->element->accept(this);
}

void CodeGenSwarm::visit(mir::WhileStmt::Ptr while_stmt) {
  if (!while_stmt->getMetadata<bool>("swarm_frontier_convert")) {
    printIndent();
    oss << "while (";
    while_stmt->cond->accept(this);
    oss << ") {" << std::endl;
    indent();
    while_stmt->body->accept(this);
    dedent();
    printIndent();
    oss << "}" << std::endl;
  } else {
    CodeGenSwarmQueueEmitter swarm_queue_emitter(oss, mir_context_, module_name);
    swarm_queue_emitter.current_while_stmt = while_stmt;
    swarm_queue_emitter.setIndentLevel(this->getIndentLevel());
    while_stmt->accept(&swarm_queue_emitter);
  }
}

void CodeGenSwarmQueueEmitter::visit(mir::WhileStmt::Ptr while_stmt) {
  // Includes checks for swarm_switch_convert and bucket queue to include switch case setup or multiple lambdas
  // if necessary.
  auto frontier_name = while_stmt->getMetadata<mir::Var>("swarm_frontier_var");
  
  if (!while_stmt->hasMetadata<bool>("update_priority_queue") || !while_stmt->getMetadata<bool>("update_priority_queue")) {
    // Flush any vertices in the frontier into the swarm Queue
    printIndent();
    oss << "for (int i = 0; i < " << frontier_name.getName() << ".size(); i++){" << std::endl;
    indent();
    printIndent();
    oss << "swarm_" <<frontier_name.getName() << ".push_init(0, ";

    if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      oss << "std::make_tuple(";
    }
    if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      for (int i = 0; i < while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size(); i++) {
        oss << "0, ";
      }
    }
    if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      oss << frontier_name.getName() << "[i]));" << std::endl;
    } else {
      oss << frontier_name.getName() << "[i]);" << std::endl;
    }
    dedent();
    printIndent();
    oss << "}" << std::endl;
  }
  printIndent();
  oss << "swarm_" << frontier_name.getName() << ".for_each_prio([";
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("global_vars")) {
    for (int i = 0; i < while_stmt->getMetadata<std::vector<mir::Var>>("global_vars").size(); i++) {
      mir::Var global_var = while_stmt->getMetadata<std::vector<mir::Var>>("global_vars")[i];
      oss << "&" << global_var.getName();
      if (i < while_stmt->getMetadata<std::vector<mir::Var>>("global_vars").size() - 1) {
        oss << ", ";
      }
    }
  }
  oss << "](unsigned level, ";
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
    oss << "std::tuple<";
  }
  frontier_name.getType()->accept(this);
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
    oss << ", ";
    for (int i = 0; i < while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size(); i++) {
      auto add_var = while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars")[i];
      add_var.getType()->accept(this);
      if (i < while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size() - 1) {
        oss << ", ";
      }
    }
    oss << ">";
  }
  oss << (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars") ? " src_tuple" : " src");
  oss << ", auto push) {" << std::endl;
  indent();
  if (while_stmt->getMetadata<bool>("swarm_switch_convert")) {
    int rounds = while_stmt->body->stmts->size();
    if (swarm_queue_type == QueueType::BUCKETQUEUE) rounds = while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_single_bucket")->stmts->size();
    printIndent();
    oss << "switch (level % " << std::to_string(rounds) << ") {" << std::endl;
  }

  if (while_stmt->hasMetadata<mir::StmtBlock::Ptr>("new_single_bucket")) {
    if (swarm_queue_type != QueueType::BUCKETQUEUE) assert(false && "Requires using bucket queue.");
    while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_single_bucket")->accept(this);
  } else {
    while_stmt->body->accept(this);
  }

  if (while_stmt->getMetadata<bool>("swarm_switch_convert")) {
    printIndent();
    oss << "}" << std::endl;
  }

  // follow up check to see whether there are frontier level switch cases that need to be processed in a separate
  // lambda. If you're using a bucket queue you always need to generate the lambda, but only search
  // for things to put in it if swarm_switch_convert is true and you have statements to put in the lambda.
  if (swarm_queue_type == QueueType::BUCKETQUEUE) {
    //if (while_stmt->getMetadata<bool>("swarm_switch_convert")) {
    //if (swarm_queue_type == QueueType::BUCKETQUEUE) {
      dedent();
      printIndent();
      oss << "}, [";
      if (while_stmt->hasMetadata<std::vector<mir::Var>>("global_vars")) {
        for (int i = 0; i < while_stmt->getMetadata<std::vector<mir::Var>>("global_vars").size(); i++) {
          mir::Var global_var = while_stmt->getMetadata<std::vector<mir::Var>>("global_vars")[i];
          oss << "&" << global_var.getName();
          if (i < while_stmt->getMetadata<std::vector<mir::Var>>("global_vars").size() - 1) {
            oss << ", ";
          }
        }
      }
      oss << "](unsigned level) {" << std::endl;
      indent();
      if (while_stmt->getMetadata<bool>("swarm_switch_convert")) {
	      int rounds = while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")->stmts->size();
	      printIndent();
	      oss << "switch (level % " << std::to_string(rounds) << ") {" << std::endl;
	      while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")->accept(this);
      	      dedent();
	      printIndent();
              oss << "}" << std::endl;
      }
  //  }
  }

  dedent();
  printIndent();
  oss << "});" << std::endl;
  if (!while_stmt->hasMetadata<bool>("update_priority_queue") || !while_stmt->getMetadata<bool>("update_priority_queue")) {
    printIndent();
    oss << "swarm_runtime::clear_frontier(" << frontier_name.getName() << ");" << std::endl;
  }
}

void CodeGenSwarm::visit(mir::ForStmt::Ptr stmt) {
  printIndent();
  oss << "for(int " << stmt->loopVar << " = ";
  stmt->domain->lower->accept(this);
  oss << "; " << stmt->loopVar << " < ";
  stmt->domain->upper->accept(this);
  oss << "; " << stmt->loopVar << "++) {" << std::endl;
  indent();
  stmt->body->accept(this);
  dedent();
  printIndent();
  oss << "}" << std::endl;

}
void CodeGenSwarm::visit(mir::ReduceStmt::Ptr stmt) {
  switch(stmt->reduce_op_) {
    case mir::ReduceStmt::ReductionOp::MIN:
      printIndent();
      if (stmt->hasMetadata<std::string>("tracking_var_name_") && stmt->getMetadata<std::string>("tracking_var_name_") != "")
        oss << stmt->getMetadata<std::string>("tracking_var_name_") << " = ";
      oss << "swarm_runtime::min_reduce(";
      stmt->lhs->accept(this);
      oss << ", ";
      stmt->expr->accept(this);
      oss << ");" << std::endl;
      break;
    case mir::ReduceStmt::ReductionOp::SUM:
      printIndent();
      if (stmt->hasMetadata<std::string>("tracking_var_name_") && stmt->getMetadata<std::string>("tracking_var_name_") != "")
        oss << stmt->getMetadata<std::string>("tracking_var_name_") << " = ";
      oss << "swarm_runtime::sum_reduce(";
      stmt->lhs->accept(this);
      oss << ", ";
      stmt->expr->accept(this);
      oss << ");" << std::endl;
      break;
  }

}

void CodeGenSwarm::visit(mir::VertexSetApplyExpr::Ptr vsae) {
  auto mir_var = mir::to<mir::VarExpr>(vsae->target);

  if (mir_context_->isConstVertexSet(mir_var->var.getName())) {
    // This is iterating over all the vertices of the graph
    auto associated_element_type = mir_context_->getElementTypeFromVectorOrSetName(mir_var->var.getName());
    auto associated_element_type_size = mir_context_->getElementCount(associated_element_type);
    oss << "for (int _iter = 0; _iter < ";
    associated_element_type_size->accept(this);
    oss << "; _iter++) {" << std::endl;
    indent();
    printIndent();
    oss << vsae->input_function_name << "(_iter);" << std::endl;
    dedent();
    printIndent();
    oss << "}";
  } else {
    oss << "for (int i = 0; i < "<<mir_var->var.getName() << ".size(); i++) {" << std::endl;
    indent();
    printIndent();
    oss << "int32_t current = " << mir_var->var.getName() << "[i];" << std::endl;
    printIndent();
    oss << vsae->input_function_name << "(current);" << std::endl;
    dedent();
    printIndent();
    oss << "}";
  }
}

void CodeGenSwarmQueueEmitter::visit(mir::StmtBlock::Ptr stmts) {
  if (stmts->stmts == nullptr) return;
  for (auto stmt: *(stmts->stmts)) {
    stmt->accept(this);

    // count how many switch cases we've gone through so far. Also to make sure a push is
    // inserted in each case.
    if (mir::isa<mir::SwarmSwitchStmt>(stmt)) {
      stmt_idx++;
      push_inserted = false;
    }
  }
}

void CodeGenSwarmQueueEmitter::visit(mir::VertexSetApplyExpr::Ptr vsae) {
  auto mir_var = mir::to<mir::VarExpr>(vsae->target);
  oss << vsae->input_function_name << "(";
  printSrcVertex();
  oss << ");" << std::endl;
}

void CodeGenSwarm::visit(mir::PushEdgeSetApplyExpr::Ptr esae) {
  auto mir_var = mir::to<mir::VarExpr>(esae->target);
  if (esae->from_func == "") {
    oss << "for (int _iter = 0; _iter < " << mir_var->var.getName() << ".num_edges; _iter++) {" << std::endl;
    indent();
    printIndent();
    oss << "int _src = " << mir_var->var.getName() << ".h_edge_src[_iter];" << std::endl;
    printIndent();
    oss << "int _dst = " << mir_var->var.getName() << ".h_edge_dst[_iter];" << std::endl;
    if (esae->is_weighted) {
      printIndent();
      oss << "int _weight = " << mir_var->var.getName() << ".h_edge_weight[_iter];" << std::endl;
    }
    printIndent();
    oss << esae->input_function_name << "(_src, _dst";
    if (esae->is_weighted)
      oss << ", _weight";
    oss << ");" << std::endl;
    dedent();
    printIndent();
    oss << "}";
  } else {
    oss << "for (int i = 0; i < "<< esae->from_func <<".size(); i++) {" << std::endl;
    indent();
    printIndent();
    oss << "int32_t current = " << esae->from_func << "[i];" << std::endl;
    printIndent();
    oss << "int32_t edgeZero = " << mir_var->var.getName() << ".h_src_offsets[current];" << std::endl;
    printIndent();
    oss << "int32_t edgeLast = " << mir_var->var.getName() << ".h_src_offsets[current+1];" << std::endl;
    printIndent();
    oss << "for (int j = edgeZero; j < edgeLast; j++) {" << std::endl;
    indent();
    printIndent();
    oss << "int ngh = " << mir_var->var.getName() << ".h_edge_dst[j];" << std::endl;
    if (esae->to_func != "") {
      printIndent();
      oss << "if (" << esae->to_func << "(ngh)) {" << std::endl;
      indent();
    }
    printIndent();
    oss << esae->input_function_name << "(current, ngh";
    if (esae->is_weighted)
      oss << ", _weight";
    oss << ");" << std::endl;
    if (esae->to_func != "") {
      dedent();
      printIndent();
      oss << "}" << std::endl;
    }
    dedent();
    printIndent();
    oss << "}" << std::endl;
    dedent();
    printIndent();
    oss << "}";
  }
}

void CodeGenSwarmQueueEmitter::visit(mir::FuncDecl::Ptr func_decl) {
  // this prints the "bool output" thing
  if (func_decl->result.isInitialized()) {
    printIndent();
    func_decl->result.getType()->accept(this);
    oss << " " << func_decl->result.getName() << ";" << std::endl;
  }

  func_decl->body->accept(this);

  // this prints the "if(output) then {do something}" thing
  if (func_decl->result.isInitialized()) {
    if (mir::isa<mir::ScalarType>(func_decl->result.getType())) {
      if (mir::to<mir::ScalarType>(func_decl->result.getType())->type == mir::ScalarType::Type::BOOL) {
        if (!push_inserted) {
		printIndent();
		oss << "if (" << func_decl->result.getName() << ") {" << std::endl;
		indent();
		printPushStatement(true, false, "dst");
		push_inserted = true;
		dedent();
		printIndent();
		oss << "}" << std::endl;
      
	}
      }
    }
  }
}

void CodeGenSwarmQueueEmitter::visit(mir::PushEdgeSetApplyExpr::Ptr esae) {
  auto mir_var = mir::to<mir::VarExpr>(esae->target);
  printIndent();
  oss << "int32_t edgeZero = " << mir_var->var.getName() << ".h_src_offsets[";
  printSrcVertex();
  oss << "];" << std::endl;
  printIndent();
  oss << "int32_t edgeLast = " << mir_var->var.getName() << ".h_src_offsets[";
  printSrcVertex();
  oss << "+1];" << std::endl;
  printIndent();
  oss << "for (int i = edgeZero; i < edgeLast; i++) {" << std::endl;
  indent();
  printIndent();
  oss << "int dst = " << mir_var->var.getName() << ".h_edge_dst[i];" << std::endl;

  if (esae->is_weighted) {
    printIndent();
    oss << "int weight = " << mir_var->var.getName() << ".h_edge_weight[i];" << std::endl;
  }

  if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
    printIndent();
    oss << "int src = ";
    printSrcVertex();
    oss << ";" << std::endl;
  }

  if (esae->to_func != "") {
    printIndent();
    oss << "if (" << esae->to_func << "(dst)) {" << std::endl;
    indent();
  }

  printIndent();
  oss << "{" << std::endl;
  indent();
  auto func_decl = mir_context_->functions_map_[esae->input_function_name];
  func_decl->accept(this);
  dedent();
  printIndent();
  oss << "}" << std::endl;

  if (esae->to_func != "") {
    dedent();
    printIndent();
    oss << "}" << std::endl;
  }

  dedent();
  printIndent();
  oss << "}" << std::endl;
}

void CodeGenSwarm::visit(mir::OrderedProcessingOperator::Ptr opo) {
  std::string pq = opo->priority_queue_name;
  printIndent();
  oss << "unsigned " << pq << "_var = 0;" << std::endl;
  printIndent();
  oss << "while (!" << pq << ".empty(" << pq << "_var)) {" << std::endl;
  indent();
  printIndent();
  oss << pq << ".for_each(" << pq << "_var, [" << pq << "_var](int node) {" << std::endl;
  indent();
  printIndent();
  oss << "for (int eid = ";
  opo->graph_name->accept(this);
  oss << ".h_src_offsets[node]; eid < ";
  opo->graph_name->accept(this);
  oss << ".h_src_offsets[node+1]; eid++) {" << std::endl;
  indent();
  printIndent();
  oss << "int neigh = ";
  opo->graph_name->accept(this);
  oss << ".h_edge_dst[eid];" << std::endl;
  printIndent();
  oss << "auto weight = ";
  opo->graph_name->accept(this);
  oss << ".h_edge_weight[eid];" << std::endl;
  printIndent();
  oss << opo->edge_update_func << "(node, neigh, weight);" << std::endl;
  dedent();
  printIndent();
  oss << "}" << std::endl;
  dedent();
  printIndent();
  oss << "});" << std::endl;
  printIndent();
  oss << pq << "_var++;" << std::endl;
  dedent();
  printIndent();
  oss << "}" << std::endl;
}

void CodeGenSwarm::visit(mir::PriorityUpdateOperatorMin::Ptr puom) {
  oss << "if (";
  puom->old_val->accept(this);
  oss << " > ";
  puom->new_val->accept(this);
  oss << ") {" << std::endl;
  indent();
  printIndent();
  puom->old_val->accept(this);
  oss << " = ";
  puom->new_val->accept(this);
  oss << ";" << std::endl;
  dedent();
  printIndent();
  oss << "}";
}

void CodeGenSwarmQueueEmitter::visit(mir::PriorityUpdateOperatorMin::Ptr puom) {
  oss << "if (";
  puom->old_val->accept(this);
  oss << " > ";
  puom->new_val->accept(this);
  oss << ") {" << std::endl;
  indent();
  printIndent();
  puom->old_val->accept(this);
  oss << " = ";
  puom->new_val->accept(this);
  oss << ";" << std::endl;
  printIndent();
  oss << "push((";
  puom->new_val->accept(this);
  oss << ")/" ;
  auto pq_var_expr = mir::to<mir::VarExpr>(puom->priority_queue);
  oss << "swarm_" << pq_var_expr->var.getName() << "_delta, ";
  puom->destination_node_id->accept(this);
  oss << ");" << std::endl;
  dedent();
  printIndent();
  oss << "}";
  push_inserted = true;
}

}
