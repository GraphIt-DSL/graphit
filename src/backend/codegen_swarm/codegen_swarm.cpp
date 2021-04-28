#include <graphit/backend/codegen_swarm/codegen_swarm.h>
#include <graphit/midend/mir.h>

namespace graphit {
int CodeGenSwarm::genSwarmCode(void) {
  std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

  for (auto it = functions.begin(); it != functions.end(); it++) {
    it->get()->accept(frontier_finder);
    it->get()->accept(dedup_finder); // find deduplication vectors from the edgesetapplyexpr metadata.
    it->get()->accept(transpose_lifter);  // lift transposes out of the swarm main function, and into the global scope.
  }

  genIncludeStmts();
  oss << "int __argc;" << std::endl;
  oss << "char **__argv;" << std::endl;

  genEdgeSets();
  genConstants();
  genSwarmStructs();
  genDedupVectors();

  for (auto it = functions.begin(); it != functions.end(); it++) {
    auto fxn_ptr = it->get();
    if (!fxn_ptr->hasMetadata<bool>("inline_only") || !fxn_ptr->getMetadata<bool>("inline_only")) {
      fxn_ptr->accept(this);
    }
  }

  genMainFunction();
  return 0;
}

int CodeGenSwarm::genIncludeStmts(void) {
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

int CodeGenSwarm::genDedupVectors(void) {
  for (auto vector_var : dedup_finder->get_vector_vars()) {
    vector_var.getType()->accept(this);
    oss << "* " << vector_var.getName() << ";" << std::endl;
  }
}

int CodeGenSwarm::genSwarmStructs() {
  for (auto const& edgeset_var : frontier_finder->edgeset_var_map) {
    std::string edgeset_name = edgeset_var.first;
    mir::VarDecl::Ptr var = edgeset_var.second;
    if (var->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      oss << "struct " << (edgeset_name + "_struct") << " {" << std::endl;
      indent();
      // the first src vertex
      printIndent();
      if (mir::isa<mir::VertexSetType>(var->type)) {
        mir::to<mir::VertexSetType>(var->type)->element->accept(this);
      } else {
        var->type->accept(this);
      }
      oss << " src;" << std::endl;
      // the rest of the additional attributes
      for (auto add_src_var : var->getMetadata<std::vector<mir::Var>>("add_src_vars")) {
        printIndent();
	add_src_var.getType()->accept(this);
	oss << " " << edgeset_name << "_" << add_src_var.getName() << ";" << std::endl;
      }
      dedent();
      oss << "};" << std::endl;
    }
  }
}

void CodeGenSwarm::printSpatialHint(mir::Expr::Ptr expr) {
        if (mir::isa<mir::TensorArrayReadExpr>(expr)) {
		auto tare = mir::to<mir::TensorArrayReadExpr>(expr);
		if (mir::isa<mir::VarExpr>(tare->target)) {
			auto name = mir::to<mir::VarExpr>(tare->target)->var.getName();
			auto index = mir::to<mir::VarExpr>(tare->index)->var.getName();
			printIndent();
			oss << "auto* hint_addr = &(" << name << "[" << index << "]);" << std::endl;
			printIndent();
			oss << "SCC_OPT_TASK();" << std::endl;
			printIndent();
			oss << "SCC_OPT_CACHELINEHINT(hint_addr);" << std::endl;
		}
	} else {
		printIndent();
		oss << "SCC_OPT_TASK();" << std::endl;
	}
}

void CodeGenSwarm::printSpatialHint() {
        printIndent();
	oss << "SCC_OPT_TASK();" << std::endl;
}

void CodeGenSwarmTransposeLifter::visit(mir::Call::Ptr call) {
  if (call->name == "builtin_transpose") {
    transpose_found = true;
  }
}

void CodeGenSwarmTransposeLifter::visit(mir::VarDecl::Ptr var_decl) {
  transpose_found = false;
  if (var_decl->initVal != nullptr)
	  var_decl->initVal->accept(this);
  if (transpose_found) transpose_decls.push_back(var_decl);
}

void CodeGenSwarmTransposeLifter::visit(mir::StmtBlock::Ptr body) {
  std::vector<mir::Stmt::Ptr> new_stmts;
  for (auto stmt : *(body->stmts)) {
    stmt->accept(this);
    if (!transpose_found) {
      new_stmts.push_back(stmt);
    }
    transpose_found = false;
  }
  (*(body->stmts)) = new_stmts;
}

void CodeGenSwarmFrontierFinder::visit(mir::VarDecl::Ptr var_decl) {
  // Add a frontier to the map once declared.
  if (mir::isa<mir::VertexSetType>(var_decl->type)) {
    std::string name = var_decl->name;
    edgeset_var_map[name] = var_decl;
  }
}

void CodeGenSwarmFrontierFinder::visit(mir::WhileStmt::Ptr while_stmt) {
  if (while_stmt->hasMetadata<bool>("swarm_frontier_convert") && while_stmt->getMetadata<bool>("swarm_frontier_convert")) {
    // HANDLES BUCKET AND PRIOQUEUE CASES bc swarm_frontier_convert is True.
    auto frontier_var = while_stmt->getMetadata<mir::Var>("swarm_frontier_var");

    // Add frontiers to the list of frontiers in the Finder object. Determine queue type from any attached schedule, additional local variables, and default to PrioQueue
    QueueType queue_type = QueueType::PRIOQUEUE;
    // if converting to switch statements, then you might need to use the BucketQueue.
    if (while_stmt->getMetadata<bool>("swarm_switch_convert")
        && while_stmt->hasMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")
            && while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")->stmts->size() > 0) {
      // BUCKETQUEUE - Check to make sure you've declared the frontier.
      if (edgeset_var_map.find(frontier_var.getName()) != edgeset_var_map.end()) {
        queue_type = QueueType::BUCKETQUEUE;
      }
    } else {
      // default PRIOQUEUE, unless schedule wants a bucketqueue.
      if (edgeset_var_map.find(frontier_var.getName()) != edgeset_var_map.end()) {
        queue_type = QueueType::PRIOQUEUE;
	    if (while_stmt->hasApplySchedule()) {
	      auto apply_schedule = while_stmt->getApplySchedule();
	      if (!apply_schedule->isComposite()) {
          auto applied_simple_schedule = apply_schedule->self<fir::swarm_schedule::SimpleSwarmSchedule>();
          if (applied_simple_schedule->queue_type == fir::swarm_schedule::SimpleSwarmSchedule::QueueType::BUCKETQUEUE) {
            queue_type = QueueType::BUCKETQUEUE;
          }
	      }
	    }
      }
    }

    swarm_frontier_vars.push_back(frontier{frontier_var.getName(), queue_type});
    
    // If there are additional local variables, store them in a map for easy access. Then store the frontier var as metadata in the edgeset_var_map
    if (edgeset_var_map.find(frontier_var.getName()) != edgeset_var_map.end()) {
      if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
        auto add_src_vars = while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars");
        edgeset_var_map[frontier_var.getName()]->setMetadata<std::vector<mir::Var>>("add_src_vars", add_src_vars);
      }
      edgeset_var_map[frontier_var.getName()]->setMetadata<mir::Var>("swarm_frontier_var", while_stmt->getMetadata<mir::Var>("swarm_frontier_var"));
    }
  } else {
   // UNORDERED QUEUE CASE
   if (while_stmt->hasApplySchedule()) {
      auto apply_schedule = while_stmt->getApplySchedule();
      if (!apply_schedule->isComposite()) {
	auto applied_simple_schedule = apply_schedule->self<fir::swarm_schedule::SimpleSwarmSchedule>();
	if (applied_simple_schedule->queue_type == fir::swarm_schedule::SimpleSwarmSchedule::QueueType::UNORDEREDQUEUE && while_stmt->hasMetadata<mir::Var>("swarm_frontier_var")) {
	  QueueType queue_type = QueueType::UNORDEREDQUEUE;
          auto frontier_var = while_stmt->getMetadata<mir::Var>("swarm_frontier_var");
	  if (edgeset_var_map.find(frontier_var.getName()) != edgeset_var_map.end()) {
            swarm_frontier_vars.push_back(frontier{frontier_var.getName(), queue_type});
	    edgeset_var_map[frontier_var.getName()]->setMetadata<mir::Var>("swarm_frontier_var", while_stmt->getMetadata<mir::Var>("swarm_frontier_var"));
	  }
	}
      }
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
      oss << "swarm::";
      if (frontier_finder->isa_frontier(constant->name)) {
        oss << frontier_finder->get_queue_string(constant->name);
      } else {
        oss << "DefaultQueue";
      }
      oss << "<int> swarm_" << constant->name << ";" << std::endl;
      oss << "int swarm_" << constant->name << "_delta;" << std::endl;
    }
  }

// Globally init all the transpose edges vars.
  for (auto var_decl : transpose_lifter->transpose_decls) {
    printIndent();
    var_decl->type->accept(this);
    oss << " " << var_decl->name << ";" << std::endl; 
  }
}

void CodeGenSwarm::visit(mir::IntLiteral::Ptr literal) {
  oss << literal->val;
}
void CodeGenSwarm::visit(mir::FloatLiteral::Ptr literal) {
  oss << "(";
  oss << "(float) ";
  oss << literal->val;
  oss << ") ";
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
  if (call->name == "builtin_sizeOf") {
    oss << "sizeof(";
    if (call->generic_type != nullptr) {
      call->generic_type->accept(this);
    } else {
      oss << "int";
    }
    oss << ")";
    return;
  }
  if (call->name.find("builtin_") == 0 || call->name == "startTimer" || call->name == "stopTimer" || call->name == "deleteObject")
    oss << "swarm_runtime::";
  oss << call->name;
  oss << "(";
  bool printDelimeter = false;
  for (auto expr: call->args) {
    if (printDelimeter)
      oss << ", ";
    expr->accept(this);
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
      if (vector_type->element_type != nullptr) {
        mir_context_->getElementCount(vector_type->element_type)->accept(this);
      } else {
        oss << std::to_string(vector_type->range_indexset);
      }  
    oss << "];" << std::endl;
    }
  }

  // Lift the transpose calls up to the main, global scope.
  for (mir::VarDecl::Ptr var_decl : transpose_lifter->transpose_decls) {
    printIndent();
    oss << var_decl->name << " = ";
    var_decl->initVal->accept(this);
    oss << ";" << std::endl;
  }
  for (auto stmt : mir_context_->field_vector_init_stmts) {
    stmt->accept(this);
  }
  printIndent();
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

void CodeGenSwarm::visit(mir::AssignStmt::Ptr stmt) {
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
  // If an edgeset is found, but was not logged in the frontierfinder, and is set to an ESAE with a known frontier, it must be an intermdiate output.
  // Allocate it into an UnorderedQueue, and then run the ESAE per usu, pushing into the intermediate output.
  if (!frontier_finder->isa_frontier(stmt->name)) {
    if (mir::isa<mir::EdgeSetApplyExpr>(stmt->initVal)) {
      auto esae = mir::to<mir::EdgeSetApplyExpr>(stmt->initVal);
      // this is like var output : vertexset{Vertex} = edges.from(frontier).applyModified(....
      if (frontier_finder->isa_frontier(esae->from_func)) {
	// Divide into two nodes: an allocation for the stmt output var, which is the same as the input frontier, and the actual ESAE for the esae->from_func var.
	auto new_output_decl = std::make_shared<mir::VarDecl>();
	new_output_decl->name = stmt->name;
	new_output_decl->type = stmt->type;
	new_output_decl->initVal = frontier_finder->edgeset_var_map[esae->from_func]->getMetadata<mir::Expr::Ptr>("alloc_expr");
	frontier_finder->swarm_frontier_vars.push_back(CodeGenSwarmFrontierFinder::frontier{stmt->name, CodeGenSwarmFrontierFinder::QueueType::UNORDEREDQUEUE});
        new_output_decl->accept(this);
	stmt->initVal->setMetadata<std::string>("output_frontier", stmt->name);

        printIndent();
	stmt->initVal->accept(this);
        return;
      } else if (esae->from_func == "") {
	// this is like var output : vertexset{Vertex] = edges.applyModified(..... 
	// Divide into two nodes: an allocation for the stmt output var, and the ESAE, which will run on the edges (constant) edgeset.
	if (mir::isa<mir::VarExpr>(esae->target)) {
	auto var_expr_target = mir::to<mir::VarExpr>(esae->target);
	if (mir_context_->isEdgeSet(var_expr_target->var.getName())) {
		auto new_output_decl = std::make_shared<mir::VarDecl>();
		new_output_decl->name = stmt->name;
		new_output_decl->type = stmt->type;
		auto new_alloc = std::make_shared<mir::VertexSetAllocExpr>();
		auto new_elem_type = std::make_shared<mir::ElementType>();
		new_elem_type->ident = "Vertex";
		new_alloc->element_type = new_elem_type;
		new_output_decl->initVal = new_alloc;
		frontier_finder->swarm_frontier_vars.push_back(CodeGenSwarmFrontierFinder::frontier{stmt->name, CodeGenSwarmFrontierFinder::QueueType::UNORDEREDQUEUE});
		new_output_decl->accept(this);
		stmt->initVal->setMetadata<std::string>("output_frontier", stmt->name);

		printIndent();
		stmt->initVal->accept(this);
		oss << ";" << std::endl;
		return;
	}
	}
      }
    }
  }
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
    oss << "swarm::" << frontier_finder->get_queue_string(stmt->name) << "<";

    // The extra types that are passed task to task
    if (stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      oss << stmt->name << "_struct";
    } else {
      oss << "int"; 
    }

    // if the frontier is not primarily defined as a Prio/BucketQueue
    if (frontier_finder->get_queue_string(stmt->name) == "UnorderedQueue") {
      oss << ">* " << stmt->name << " = new swarm::UnorderedQueue<";
      oss << "int";
      oss << ">();" << std::endl;
      // Produce the initialization of the UnorderedQueue with the given size if its a VertexSetAlloc. Also store the initial vertices to push, optionally, which is later handled in the WhileStmt.
      if (mir::isa<mir::VertexSetAllocExpr>(stmt->initVal)) {
	auto vsae = mir::to<mir::VertexSetAllocExpr>(stmt->initVal);
        printIndent();
	oss << stmt->name << "->init(";
        mir_context_->getElementCount(vsae->element_type)->accept(this);
	oss << ");" << std::endl;
	if (frontier_finder->edgeset_var_map.find(stmt->name) == frontier_finder->edgeset_var_map.end()) {
	  frontier_finder->edgeset_var_map[stmt->name] = stmt;
	}
        frontier_finder->edgeset_var_map[stmt->name]->setMetadata<mir::Expr::Ptr>("alloc_expr", vsae);
      }
      return; 
    } else {
      oss << "> swarm_" << stmt->name << ";" << std::endl;
      
      // For Prio/Bucketqueue, we want to initialize a second UnorderedQueue representation of the frontier.
      printIndent();
      oss << "swarm::UnorderedQueue<int>";

      oss << "* " << stmt->name << " = new swarm::UnorderedQueue<";
      oss << "int";
      oss << ">();" << std::endl;
      // Produce the initialization of the UnorderedQueue with the given size if its a VertexSetAlloc. Also store the initial vertices to push, optionally, which is later handled in the WhileStmt.
      if (mir::isa<mir::VertexSetAllocExpr>(stmt->initVal)) {
	auto vsae = mir::to<mir::VertexSetAllocExpr>(stmt->initVal);
        printIndent();
	oss << stmt->name << "->init(";
        mir_context_->getElementCount(vsae->element_type)->accept(this);
	oss << ");" << std::endl;
	if (frontier_finder->edgeset_var_map.find(stmt->name) == frontier_finder->edgeset_var_map.end()) {
	  frontier_finder->edgeset_var_map[stmt->name] = stmt;
	}
        frontier_finder->edgeset_var_map[stmt->name]->setMetadata<mir::Expr::Ptr>("alloc_expr", vsae);
      }
      return;
    }
  }

  // Default initialization of other non-Edgeset, non-frontier variables.
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
  CodeGenSwarm::visit(stmt);
}

void CodeGenSwarmQueueEmitter::visit(mir::SwarmSwitchStmt::Ptr switch_stmt) {
  printIndent();
  oss << "case " << std::to_string(switch_stmt->round) << ": {" << std::endl;
  indent();

  // print a push unless you're at the last frontier level stmt.
  if (!switch_stmt->getMetadata<bool>("is_vertex_level") && current_while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")->stmts->size() - 1 == stmt_idx) {
    push_inserted = true;
  }
  
  switch_stmt->stmt_block->accept(this);
  if (switch_stmt->getMetadata<bool>("is_vertex_level")) {
    if (!push_inserted) {
      // is_insert_call is analogous to whether we need to increment the VFL round or not.
      // enq same vertex.
      printPushStatement(is_insert_call, true);
      is_insert_call = false;
    }
  } else {
    if (!push_inserted) {
      printPushStatement(is_insert_call, true);
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
  if (call->name == "builtin_insert" || call->name == "builtin_update_size") {
    if (call->hasMetadata<mir::Var>("increment_round_var")) {
      is_insert_call = true;
    }
  }
  if (call->name == "builtin_sizeOf") {
    oss << "sizeof(";
    if (call->generic_type != nullptr) {
      call->generic_type->accept(this);
    } else {
      oss << "int";
    }
    oss << ")";
    return;
  }
  if (call->name.find("builtin_") == 0 || call->name == "startTimer" || call->name == "stopTimer" || call->name == "deleteObject")
    oss << "swarm_runtime::";
  oss << call->name;
  oss << "(";
  bool printDelimeter = false;
  for (auto expr: call->args) {
    if (printDelimeter)
      oss << ", ";
    expr->accept(this);
    printDelimeter = true;
  }
  if (call->hasMetadata<mir::Var>("increment_round_var")) {
    oss << ", ";
    printRoundIncrement(call->getMetadata<mir::Var>("increment_round_var").getName());
    if (call->name == "builtin_update_size") {
      oss << " + 1";
    }
  }
  oss << ")";
}


void CodeGenSwarmQueueEmitter::visit(mir::VarExpr::Ptr expr) {
  if (expr->var.getName() == current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName())
    printSrcVertex();
  else
    CodeGenSwarm::visit(expr);
}

void CodeGenSwarmQueueEmitter::visit(mir::VertexSetType::Ptr vertexset_type) {
  vertexset_type->element->accept(this);
}

void CodeGenSwarm::visit(mir::WhileStmt::Ptr while_stmt) {
  // Either UnorderedQueue or just a regular While loop. regular while loops won't have the swarm_frontier_var metadata set, because they're not iterating on any frontiers.
  if (!while_stmt->hasMetadata<bool>("swarm_frontier_convert") || !while_stmt->getMetadata<bool>("swarm_frontier_convert") || !while_stmt->hasMetadata<mir::Var>("swarm_frontier_var")) {
    // UNORDEREDQUEUE CASE must push initial vertices before looping in the while loop.
    if (while_stmt->hasMetadata<mir::Var>("swarm_frontier_var")) {
	auto frontier_name = while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName();
        // Initialize vertices in the UnorderedQueue case, if there are vertices to initialize.
	if (frontier_finder->edgeset_var_map[frontier_name]->hasMetadata<mir::Expr::Ptr>("alloc_expr")) {
		auto vsae =  mir::to<mir::VertexSetAllocExpr>(frontier_finder->edgeset_var_map[frontier_name]->getMetadata<mir::Expr::Ptr>("alloc_expr"));
		if (vsae->size_expr != nullptr) {
			printIndent();
			oss << "for (int i = 0, m = ";
			vsae->size_expr->accept(this);
			oss << "; i < m; i++) {" << std::endl;
			indent();
			printIndent();
			oss << frontier_name << "->push(i);" << std::endl;
			dedent();
			printIndent();
			oss << "}" << std::endl;
		}
	}
    }
    beginDedupSection(while_stmt);
    printIndent();
    oss << "while (";
    while_stmt->cond->accept(this);
    oss << ") {" << std::endl;
    indent();
    while_stmt->body->accept(this);
    dedent();
    printIndent();
    oss << "}" << std::endl;
    endDedupSection(while_stmt);
  } else {
    // PRIO OR BUCKET QUEUE CASE
    CodeGenSwarmQueueEmitter swarm_queue_emitter(oss, mir_context_, module_name, dedup_finder);
    swarm_queue_emitter.current_while_stmt = while_stmt;
    swarm_queue_emitter.setIndentLevel(this->getIndentLevel());
    if (frontier_finder->get_queue_string(while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName()) == "BucketQueue") {
      swarm_queue_emitter.swarm_queue_type = CodeGenSwarmQueueEmitter::QueueType::BUCKETQUEUE;
    } 
    while_stmt->accept(&swarm_queue_emitter);
  }
}

void CodeGenSwarmDedupFinder::visit(mir::WhileStmt::Ptr while_stmt) {
  enclosing_while_stmt = while_stmt;
  while_stmt->cond->accept(this);
  while_stmt->body->accept(this);
  enclosing_while_stmt = nullptr;
}

void CodeGenSwarmDedupFinder::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
  if (pesae->hasMetadata<mir::Var>("dedup_vector")) {
    if (enclosing_while_stmt != nullptr) {
      std::vector<mir::Var> dedup_vectors;
      if (enclosing_while_stmt->hasMetadata<std::vector<mir::Var>>("dedup_vectors")) {
        dedup_vectors = enclosing_while_stmt->getMetadata<std::vector<mir::Var>>("dedup_vectors");
      }
      dedup_vectors.push_back(pesae->getMetadata<mir::Var>("dedup_vector"));
      enclosing_while_stmt->setMetadata<std::vector<mir::Var>>("dedup_vectors", dedup_vectors);
    }
    vector_vars.push_back(pesae->getMetadata<mir::Var>("dedup_vector"));
    mir::Call::Ptr size_call = std::make_shared<mir::Call>();
    size_call->name = "builtin_getVertices";
    size_call->args.push_back(pesae->target);
    vector_size_calls_map[pesae->getMetadata<mir::Var>("dedup_vector").getName()] = size_call;
  }
}

void CodeGenSwarm::beginDedupSection(mir::WhileStmt::Ptr while_stmt) {
  // check deduplication vectors "inFrontier"s to allocate them.
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("dedup_vectors")) {
    auto vectors_to_allocate = while_stmt->getMetadata<std::vector<mir::Var>>("dedup_vectors");
    for (int i = 0; i < vectors_to_allocate.size(); i++) {
      printIndent();
      oss << vectors_to_allocate[i].getName() << " = new ";
      vectors_to_allocate[i].getType()->accept(this);
      oss << " [";
      std::string vector_name = vectors_to_allocate[i].getName();
      auto size_map = dedup_finder->vector_size_calls_map;
      auto size_expr = size_map[vector_name];
      auto size_expr_2 = dedup_finder->get_size_call(vector_name);
      size_expr->accept(this);
      oss << "]();" << std::endl;
    }
  }
}

void CodeGenSwarm::beginDedupSection(mir::PushEdgeSetApplyExpr::Ptr pesae) {
  auto func_decl = mir_context_->functions_map_[pesae->input_function_name];
  // temporarily attach metadata to the function to produce the dedup vector if block.
  if (pesae->hasMetadata<mir::Var>("dedup_vector")) {
    func_decl->setMetadata<mir::Var>("dedup_vector", pesae->getMetadata<mir::Var>("dedup_vector"));
  }
}

void CodeGenSwarm::endDedupSection(mir::PushEdgeSetApplyExpr::Ptr pesae) {
  auto func_decl = mir_context_->functions_map_[pesae->input_function_name];
  if (pesae->hasMetadata<mir::Var>("dedup_vector")) {
    func_decl->setMetadata<mir::Var>("dedup_vector", mir::Var("", std::make_shared<mir::ScalarType>()));
  }
}

void CodeGenSwarm::beginDedupSection(mir::EnqueueVertex::Ptr enqueue_vertex) {
  std::string dst = mir::to<mir::VarExpr>(enqueue_vertex->vertex_id)->var.getName();
  // the if(inFrontier)... part
  if (enqueue_vertex->hasMetadata<mir::Var>("dedup_vector")) {
    printIndent();
    oss << "if (!" << enqueue_vertex->getMetadata<mir::Var>("dedup_vector").getName() << "[" << dst << "]) {" <<std::endl;
    indent();
    printIndent();
    oss << enqueue_vertex->getMetadata<mir::Var>("dedup_vector").getName() << "[" << dst << "] = ";
    if (mir::to<mir::ScalarType>(enqueue_vertex->getMetadata<mir::Var>("dedup_vector").getType())->type == mir::ScalarType::Type::BOOL) {
      oss << "true";
    }
    oss << ";" << std::endl;
  }
}

void CodeGenSwarm::endDedupSection(mir::EnqueueVertex::Ptr enqueue_vertex) {
  if (enqueue_vertex->hasMetadata<mir::Var>("dedup_vector")) {
    dedent();
    printIndent();
    oss << "}" << std::endl;
  }
}

void CodeGenSwarm::endDedupSection(mir::WhileStmt::Ptr while_stmt) {
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("dedup_vectors")) {
    // deallocate deduplication vectors "inFrontiers"
    auto vectors_to_allocate = while_stmt->getMetadata<std::vector<mir::Var>>("dedup_vectors");
    for (int i = 0; i < vectors_to_allocate.size(); i++) {
      printIndent();
      oss << "delete[] " << vectors_to_allocate[i].getName() << ";" << std::endl;
    }
  }
}

void CodeGenSwarmQueueEmitter::cleanUpForEachPrioCall(mir::WhileStmt::Ptr while_stmt) {
  auto frontier_name = while_stmt->getMetadata<mir::Var>("swarm_frontier_var");
  printIndent();
  oss << "});" << std::endl;
  endDedupSection(while_stmt);
  if (!while_stmt->hasMetadata<bool>("update_priority_queue") || !while_stmt->getMetadata<bool>("update_priority_queue")) {
    printIndent();
    oss << "swarm_runtime::clear_frontier(" << frontier_name.getName() << ");" << std::endl;
  }
}

void CodeGenSwarmQueueEmitter::prepForEachPrioCall(mir::WhileStmt::Ptr while_stmt) {
  // Includes checks for swarm_switch_convert and bucket queue to include switch case setup or multiple lambdas
  // if necessary.
  auto frontier_name = while_stmt->getMetadata<mir::Var>("swarm_frontier_var");
  
  if (!while_stmt->hasMetadata<bool>("update_priority_queue") || !while_stmt->getMetadata<bool>("update_priority_queue")) {
    // Flush any vertices in the UnorderedQueue version of the frontier into the swarm Prio/BucketQueue. You don't need to do this for pq
    printIndent();
    oss << frontier_name.getName() << "->for_each([&](int src) {" << std::endl;
    indent();
    printIndent();
    oss << "swarm_" <<frontier_name.getName() << ".push_init(0, ";

  if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) { // option convert type to struct
	  oss << frontier_name.getName() << "_struct{";
  }
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
    oss << "src";
  } else {
    oss << "src);" << std::endl;
  }

  // if struct, then produce initial states of the additional variables to be passed in the struct.
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
    assert(while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size() == while_stmt->getMetadata<std::vector<mir::Expr::Ptr>>("init_expr_stmts").size());
    for (int i = 0; i < while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size(); i++) {
      oss << ", ";
      while_stmt->getMetadata<std::vector<mir::Expr::Ptr>>("init_expr_stmts")[i]->accept(this);
    }
    oss << "});" << std::endl;
  }
  dedent();
  printIndent();
  oss << "});" << std::endl;
  }
  
  // check deduplication vectors "inFrontier"s to allocate them.
  beginDedupSection(while_stmt);
}

void CodeGenSwarmQueueEmitter::visit(mir::WhileStmt::Ptr while_stmt) {
  // Check if the while loop is a swarm frontier. In the case of while loops that happen inside while loops, we might run into a situation where
  // we are in the QueueEmitter but we find non-frontier while loops inside larger frontier while loops. (e.g. CC)
  if (!while_stmt->hasMetadata<mir::Var>("swarm_frontier_var") || !while_stmt->getMetadata<bool>("swarm_frontier_convert")) {
    CodeGenSwarm::visit(while_stmt);
    return;
  }

  prepForEachPrioCall(while_stmt);

  auto frontier_name = while_stmt->getMetadata<mir::Var>("swarm_frontier_var");
  // Now produce code for the for_each_prio lambda
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
  if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) { // option struct
    oss << frontier_name.getName() << "_struct";
  } else {
    frontier_name.getType()->accept(this); // else just the type of the vertex
  }
  oss << (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars") ? " src_struct" : " src");
  oss << ", auto push) {" << std::endl;
  indent();

  // optionally produce switch statement structure
  if (while_stmt->getMetadata<bool>("swarm_switch_convert")) {
    int rounds = while_stmt->body->stmts->size();
    if (swarm_queue_type == QueueType::BUCKETQUEUE) rounds = while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_single_bucket")->stmts->size();
    printIndent();
    oss << "switch (level % " << std::to_string(rounds) << ") {" << std::endl;
  }

  // Produce first lambda with tasks that process single vertices.
  if (while_stmt->hasMetadata<mir::StmtBlock::Ptr>("new_single_bucket")) {
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
      oss << "](unsigned level";
      
      if (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) { // option struct
        oss << ", " << frontier_name.getName() << "_struct";
      } else {
        oss << ", ";
	frontier_name.getType()->accept(this);  // option just type of vertex
      }
      
      oss << (while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars") ? " src_struct" : " src");

      oss << ", auto push) {" << std::endl;
      indent();
      if (while_stmt->getMetadata<bool>("swarm_switch_convert")) {
	      int rounds = while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")->stmts->size();
	      printIndent();
	      oss << "switch (level % " << std::to_string(rounds) << ") {" << std::endl;
  	      stmt_idx = 0;
	      while_stmt->getMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket")->accept(this);
	      printIndent();
              oss << "}" << std::endl;
      }
  }

  dedent();
  cleanUpForEachPrioCall(while_stmt);
}

void CodeGenSwarm::visit(mir::ForStmt::Ptr stmt) {
  printIndent();
  oss << "for(int " << stmt->loopVar << " = ";
  stmt->domain->lower->accept(this);
  oss << ", m = ";
  stmt->domain->upper->accept(this);
  oss << "; " << stmt->loopVar << " < m";
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
    case mir::ReduceStmt::ReductionOp ::ATOMIC_SUM:
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
    oss << "for (int _iter = 0, m = ";
    associated_element_type_size->accept(this);
    oss << "; _iter < m; _iter++) {" << std::endl;

    if (vsae->hasMetadata<bool>("inline_function") && vsae->getMetadata<bool>("inline_function")) {
      oss << "int v = _iter;" << std::endl;
      printIndent();
      oss << "{" << std::endl;
      indent();
      mir_context_->getFunction(vsae->input_function_name)->body->accept(this);
      dedent();
      printIndent();
      oss << "}" << std::endl;
    } else {
      indent();
      printIndent();
      oss << vsae->input_function_name << "(_iter);" << std::endl;
      dedent();
    }

    printIndent();
    oss << "}";
  } else {
	  // Iterate on frontier vertexsets
	  if (frontier_finder->isa_frontier(mir_var->var.getName())) {
	  	oss << mir_var->var.getName() << "->for_each([](int v) { " << std::endl;
		indent();
	  } else {
		  // Iterate on non-frontier vertexsets
    		oss << "for (int i = 0, m = " << mir_var->var.getName() << ".size(); i < m; i++) {" << std::endl;
		indent();
    	}
    if (vsae->hasMetadata<bool>("inline_function") && vsae->getMetadata<bool>("inline_function")) {
      // Index into a normal VertexSet (non-frontier)
      if (!(frontier_finder->isa_frontier(mir_var->var.getName()))) {
        oss << "int v = " << mir_var->var.getName() << "[i];" << std::endl;
      }
      printIndent();
      oss << "{" << std::endl;
      indent();
      mir_context_->getFunction(vsae->input_function_name)->body->accept(this);
      dedent();
      printIndent();
      oss << "}" << std::endl;
    } else {
      if (!(frontier_finder->isa_frontier(mir_var->var.getName()))) {
        printIndent();
        oss << "int32_t v = " << mir_var->var.getName() << "[i];" << std::endl;
      }
      printIndent();
      oss << vsae->input_function_name << "(v);" << std::endl;
    }
    if (frontier_finder->isa_frontier(mir_var->var.getName())) {
      dedent();
      printIndent();
      oss << "})";
    } else {
      dedent();
      printIndent();
      oss << "}";
    }
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
  if (mir_context_->isConstVertexSet(mir_var->var.getName())) {
    // This is iterating over all the vertices of the graph. Same behavior expected as VSAE outside the swarm queue loop.
    CodeGenSwarm::visit(vsae);
  } else {
    if (vsae->hasMetadata<bool>("inline_function") && vsae->getMetadata<bool>("inline_function")) {
      oss << "int v = ";
      printSrcVertex();
      oss << ";" << std::endl;
      printIndent();
      oss << "{" << std::endl;
      indent();
      mir_context_->getFunction(vsae->input_function_name)->body->accept(this);
      dedent();
      printIndent();
      oss << "}" << std::endl;
    } else {
      oss << vsae->input_function_name << "(";
      printSrcVertex();
      oss << ");" << std::endl;
    }
  }
}

void CodeGenSwarm::visit(mir::PushEdgeSetApplyExpr::Ptr esae) {
  auto mir_var = mir::to<mir::VarExpr>(esae->target);
  if (esae->from_func == "") { // PAGERANK ESAE case
    oss << "for (int _iter = 0, m = " << mir_var->var.getName() << ".num_edges; _iter < m; _iter++) {" << std::endl;
    indent();
    printIndent();
    oss << "int _src = " << mir_var->var.getName() << ".h_edge_src[_iter];" << std::endl;
    printIndent();
    oss << "int _dst = " << mir_var->var.getName() << ".h_edge_dst[_iter];" << std::endl;
    if (esae->is_weighted) {
      printIndent();
      oss << "int _weight = " << mir_var->var.getName() << ".h_edge_weight[_iter];" << std::endl;
    }
    if (esae->hasMetadata<std::string>("output_frontier") && frontier_finder->isa_frontier(esae->getMetadata<std::string>("output_frontier")) && frontier_finder->get_queue_string(esae->getMetadata<std::string>("output_frontier")) == "UnorderedQueue") {
          printIndent();
	  oss << "{" << std::endl;
	  indent();
	  printIndent();
	  oss << "int src = _src;" <<std::endl;
	  printIndent();
	  oss << "int dst = _dst;" << std::endl;
	  if (esae->hasMetadata<std::string>("output_frontier")) {
	  	printIndent();
	  	oss << "swarm::UnorderedQueue<int>* __output_frontier = " << esae->getMetadata<std::string>("output_frontier") << ";" << std::endl;
	  }
	  auto func_decl = mir_context_->functions_map_[esae->input_function_name];
	  // temporarily attach metadata to the function to produce the dedup vector if block.
	  beginDedupSection(esae);
	  inlineFunction(func_decl);
	  endDedupSection(esae);
	  dedent();
	  printIndent();
	  oss << "}" << std::endl;
    } else {
    	printIndent();
    	oss << esae->input_function_name << "(_src, _dst";
    	if (esae->is_weighted)
      	  oss << ", _weight";
    	oss << ");" << std::endl;
    }
    dedent();
    printIndent();
    oss << "}";
  } else {
    // UnorderedQueue ESAE case
    // Now let's do this for any frontier, not just unordered queue.
    if (frontier_finder->isa_frontier(esae->from_func)) {
      oss << esae->from_func << "->for_each([";
      if (esae->hasMetadata<std::string>("output_frontier")) {
	      oss << esae->getMetadata<std::string>("output_frontier");
      }
      oss << "](int src) {" << std::endl;
      indent();
    } else {
	    // Now this case is for non-frontier from_func's.
      oss << "for (int i = 0, m = " << esae->from_func << ".size(); i < m; i++) {" << std::endl;
      indent();
      printIndent();
      oss << "int32_t src = " << esae->from_func << "[i];" << std::endl;
    }
    printIndent();
    oss << "int32_t edgeZero = " << mir_var->var.getName() << ".h_src_offsets[src];" << std::endl;
    printIndent();
    oss << "int32_t edgeLast = " << mir_var->var.getName() << ".h_src_offsets[src+1];" << std::endl;
    if (esae->hasMetadata<mir::Expr::Ptr>("swarm_coarsen_expr")) {
      printIndent();
      oss << "SCC_OPT_LOOP_COARSEN_FACTOR(";
      esae->getMetadata<mir::Expr::Ptr>("swarm_coarsen_expr")->accept(this);
      oss << ")" << std::endl;
    }
    printIndent();
    oss << "for (int j = edgeZero; j < edgeLast; j++) {" << std::endl;
    indent();
    printIndent();
    oss << "int dst = " << mir_var->var.getName() << ".h_edge_dst[j];" << std::endl;

    if (esae->hasMetadata<mir::Expr::Ptr>("swarm_coarsen_expr") || esae->hasMetadata<mir::Expr::Ptr>("spatial_hint")) {
      if (esae->hasMetadata<mir::Expr::Ptr>("spatial_hint")) {
        mir::VarExpr::Ptr index_expr = std::make_shared<mir::VarExpr>();
        mir::ScalarType::Ptr scalar_type = std::make_shared<mir::ScalarType>();
        mir::Var index_var = mir::Var("dst", scalar_type);
        index_expr->var = index_var;
        
        mir::to<mir::TensorArrayReadExpr>(esae->getMetadata<mir::Expr::Ptr>("spatial_hint"))->index = index_expr;
        printSpatialHint(esae->getMetadata<mir::Expr::Ptr>("spatial_hint"));
      } else {
        printSpatialHint();
      }
    }
    if (esae->to_func != "") {
      printIndent();
      oss << "if (" << esae->to_func << "(dst)) {" << std::endl;
      indent();
    }
    if (frontier_finder->isa_frontier(esae->from_func)) {
          // For frontiers, generate this code assuming the second frontier is an UnorderedQueue
	  printIndent();
	  oss << "{" << std::endl;
	  indent();
	  if (esae->hasMetadata<std::string>("output_frontier")) {
	  	printIndent();
	  	oss << "swarm::UnorderedQueue<int>* __output_frontier = " << esae->getMetadata<std::string>("output_frontier") << ";" << std::endl;
	  }
	  auto func_decl = mir_context_->functions_map_[esae->input_function_name];
	  // temporarily attach metadata to the function to produce the dedup vector if block.
	  beginDedupSection(esae);
	  inlineFunction(func_decl);
	  endDedupSection(esae);

	  dedent();
	  printIndent();
	  oss << "}" << std::endl;
    } else {
      printIndent();
      oss << esae->input_function_name << "(src, dst";
      if (esae->is_weighted)
        oss << ", _weight";
      oss << ");" << std::endl;
    }
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
    if (frontier_finder->isa_frontier(esae->from_func)) {
      oss << ");" << std::endl;
    }
  }
}

void CodeGenSwarm::inlineFunction(mir::FuncDecl::Ptr func_decl) {
  // this prints the "bool output" thing
  if (func_decl->result.isInitialized()) {
    printIndent();
    func_decl->result.getType()->accept(this);
    oss << " " << func_decl->result.getName() << ";" << std::endl;
  }

  func_decl->body->accept(this);
  
  // this prints the "if(output) then {do something}" thing
  if (func_decl->result.isInitialized()) {
    assert(false && "This should have been produced by the EnqueueVertex.....?");
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
  if (esae->hasMetadata<mir::Expr::Ptr>("swarm_coarsen_expr")) {
    printIndent();
    oss << "SCC_OPT_LOOP_COARSEN_FACTOR(";
    esae->getMetadata<mir::Expr::Ptr>("swarm_coarsen_expr")->accept(this);
    oss << ")" << std::endl;
  }
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

  if (esae->hasMetadata<mir::Expr::Ptr>("swarm_coarsen_expr") || esae->hasMetadata<mir::Expr::Ptr>("spatial_hint")) {
    if (esae->hasMetadata<mir::Expr::Ptr>("spatial_hint")) {
      mir::VarExpr::Ptr index_expr = std::make_shared<mir::VarExpr>();
      mir::ScalarType::Ptr scalar_type = std::make_shared<mir::ScalarType>();
      mir::Var index_var = mir::Var("dst", scalar_type);
      index_expr->var = index_var;
      
      mir::to<mir::TensorArrayReadExpr>(esae->getMetadata<mir::Expr::Ptr>("spatial_hint"))->index = index_expr;
      printSpatialHint(esae->getMetadata<mir::Expr::Ptr>("spatial_hint"));
    } else {
      printSpatialHint();
    }
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
  // temporarily attach metadata to the function to produce the dedup vector if block.
  beginDedupSection(esae);
  inlineFunction(func_decl);
  func_decl->accept(this);
  endDedupSection(esae);
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

// This will not produce anything right now, probably need to push into the enqueue vertex frontier,
// but not implemented yet bc nothing is using it, as most frontier pushes occur in the SwarmQueueEmitter while loops.
void CodeGenSwarm::visit(mir::EnqueueVertex::Ptr enqueue_vertex) {
	// the if(inFrontier)... part
    beginDedupSection(enqueue_vertex);

	// pushing a vertex.
    printIndent();
    oss << "swarm_runtime::builtin_addVertex(";
    enqueue_vertex->vertex_frontier->accept(this);
    oss << ", ";
    oss << mir::to<mir::VarExpr>(enqueue_vertex->vertex_id)->var.getName() << ");" << std::endl;

    endDedupSection(enqueue_vertex);
}

// handles enqueueVertex's in while loops, where push statements must be inserted.
// Checks to see if there is a push already inserted, as some cases insert pushes manually (e.g. min reductions)
void CodeGenSwarmQueueEmitter::visit(mir::EnqueueVertex::Ptr enqueue_vertex) {
	// the if(inFrontier)... part
	beginDedupSection(enqueue_vertex);

	// the push(level.....src) part
    std::string dst = mir::to<mir::VarExpr>(enqueue_vertex->vertex_id)->var.getName();
    printPushStatement(false, false, dst);
    push_inserted = true;

	endDedupSection(enqueue_vertex);
}
}
