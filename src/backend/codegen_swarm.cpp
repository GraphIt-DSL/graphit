#include <graphit/backend/codegen_swarm.h>
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
		it->get()->accept(this);
	}

	genMainFunction();
}

int CodeGenSwarm::genIncludeStmts(void) {
	oss << "#include \"swarm_intrinsics.h\"" << std::endl;
}
int CodeGenSwarm::genEdgeSets(void) {
	for (auto edgeset : mir_context_->getEdgeSets()) {	
		auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
		edge_set_type->accept(this);
		oss << " " << edgeset->name << ";" << std::endl;
	}
}


void CodeGenSwarm::visit(mir::EdgeSetType::Ptr edge_set_type) {
	if (edge_set_type->weight_type == nullptr) {
		oss << "swarm_runtime::GraphT<char>";
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
			oss << "pls::MultiQueue<int> " << constant->name << ";" << std::endl;
			oss << "int " << constant->name << "_delta;" << std::endl;
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
	oss << "pls::enqueue(swarm_main, 0, NOHINT);" << std::endl;
	printIndent();
	oss << "pls::run();" << std::endl;
	
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
	
void CodeGenSwarm::visit(mir::AssignStmt::Ptr stmt) {
	if (mir::isa<mir::PriorityQueueAllocExpr>(stmt->expr)) {
		mir::PriorityQueueAllocExpr::Ptr pqae = mir::to<mir::PriorityQueueAllocExpr>(stmt->expr);
		auto associated_element_type_size = mir_context_->getElementCount(pqae->element_type);
		printIndent();
		stmt->lhs->accept(this);
		oss << ".init(";
		associated_element_type_size->accept(this);
		oss << ");" << std::endl;
		printIndent();
		stmt->lhs->accept(this);
		oss << "_delta = ";
		if (pqae->delta > 0) {
			oss << pqae->delta;
		} else {
			oss << "atoi(__argv[" << -pqae->delta << "])";
		}
		oss << ";" << std::endl;
		printIndent();
		stmt->lhs->accept(this);
		oss << ".push(0, ";
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
		oss << "pls::Timestamp";
	}
	oss << ") {" << std::endl;
	indent();
	func->body->accept(this);
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

void CodeGenSwarm::visit(mir::PrintStmt::Ptr stmt) {
	printIndent();
	oss << "swarm_runtime::print(";
	stmt->expr->accept(this);
	oss << ");" << std::endl;
}

void CodeGenSwarm::visit(mir::VarDecl::Ptr stmt) {
	printIndent();
	stmt->type->accept(this);
	oss << " " << stmt->name;
	if (stmt->initVal != nullptr) {
		oss << " = ";
		stmt->initVal->accept(this);
	}
	oss << ";" << std::endl;
	
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
			if (stmt->tracking_var_name_ != "")
				oss << stmt->tracking_var_name_ << " = ";
			oss << "swarm_runtime::min_reduce(";
			stmt->lhs->accept(this);
			oss << ", ";
			stmt->expr->accept(this);
			oss << ");" << std::endl;
			break;
		case mir::ReduceStmt::ReductionOp::SUM:
			printIndent();
			if (stmt->tracking_var_name_ != "")
				oss << stmt->tracking_var_name_ << " = ";
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
		assert(false && "Swarm backend doesn't support vertex set apply on frontiers\n");
	}	
}

void CodeGenSwarm::visit(mir::PushEdgeSetApplyExpr::Ptr esae) {
        auto mir_var = mir::to<mir::VarExpr>(esae->target);
        if (esae->from_func == "") {
		oss << "for (int _iter = 0; _iter < " << mir_var->var.getName() << ".num_edges; _iter++) {" << std::endl;
		indent();
		printIndent();
		oss << "int _src = " << mir_var->var.getName() << ".edge_src[_iter];" << std::endl;
		printIndent();
		oss << "int _dst = " << mir_var->var.getName() << ".edge_dst[_iter];" << std::endl;
		if (esae->is_weighted) {
			printIndent();
			oss << "int _weight = " << mir_var->var.getName() << ".edge_weight[_iter];" << std::endl;
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
		assert(false && "Swarm backend doesn't support edge set apply on frontiers directly\n");
	}
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
	oss << ".src_offsets[node]; eid < "; 
	opo->graph_name->accept(this);
	oss << ".src_offsets[node+1]; eid++) {" << std::endl;
	indent();
	printIndent();
	oss << "int neigh = ";
	opo->graph_name->accept(this);
	oss << ".edge_dst[eid];" << std::endl;
	printIndent();
	oss << "auto weight = ";
	opo->graph_name->accept(this);
	oss << ".edge_weight[eid];" << std::endl;
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
	printIndent();
	puom->priority_queue->accept(this);
	oss << ".push((";
	puom->new_val->accept(this);
	oss << ")/" ;
	puom->priority_queue->accept(this);
	oss << "_delta, ";
	puom->destination_node_id->accept(this);
	oss << ");" << std::endl;
	dedent();
	printIndent();
	oss << "}";
}

}