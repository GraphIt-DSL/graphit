#include "graphit/backend/codegen_gunrock/codegen_gunrock.h"

namespace graphit {

	void CodeGenGunrock::printIndent(void) {
		oss << std::string(indent_value, '\t');
	}
	int CodeGenGunrock::indent(void) {
		return indent_value++;
	}
	int CodeGenGunrock::dedent(void) {
		return indent_value--;
	}

	int CodeGenGunrock::genGunrockCode(void) {
		genIncludeStmts();
		genEdgeSets();

		for (auto constant : mir_context_->getLoweredConstants()) {
			if (auto type = std::dynamic_pointer_cast<mir::VectorType>(constant->type)){
				genPropertyArrayDecl(constant);
			}
		}

		// Process all functions 	
		const std::vector<mir::FuncDecl::Ptr> &functions = mir_context_->getFunctionList();	

		for (auto it = functions.begin(); it != functions.end(); it++) {
			it->get()->accept(this);
		}
		return 0;
	}

	int CodeGenGunrock::genPropertyArrayDecl(mir::VarDecl::Ptr var_decl) {
		oss << "// " << var_decl->name << std::endl;

		auto vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
		assert(vector_type != nullptr);
		auto vector_element_type = vector_type->vector_element_type;

		if (mir::isa<mir::VectorType>(vector_element_type)) {
			std::cerr << "Vector of vector not yet supported for GPU backend. Exiting with errors\n";
			exit(-1);
		}	


		oss << "util::Array1D<SizeT, ";
		vector_element_type->accept(this);
		oss << "> " << var_decl->name << ";" << std::endl;

		return 0;

	}


	int CodeGenGunrock::genPropertyArrayAlloca(mir::VarDecl::Ptr var_decl) {
		auto vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
		assert(vector_type != nullptr);
		auto size_expr = mir_context_->getElementCount(vector_type->element_type);
		assert(size_expr != nullptr);

		printIndent();	
		oss << var_decl->name << ".Allocate(";
		size_expr->accept(this);
		oss << ", util::DEVICE);" << std::endl;

		printIndent();	
		oss << var_decl->name << ".Allocate(";
		size_expr->accept(this);
		oss << ", util::HOST);" << std::endl;

		return 0;	
	}



	void CodeGenGunrock::visit(mir::Call::Ptr call_expr) {
		oss << call_expr->name << "(";
		bool print_delimeter = false;
		for (auto arg: call_expr->args) {
			if (print_delimeter)
				oss << ", ";
			arg->accept(this);
			print_delimeter = true;
		}
		oss << ")";
	}

	void CodeGenGunrock::visit(mir::VarExpr::Ptr var_expr) {
		oss << var_expr->var.getName();
	}

	void CodeGenGunrock::visit(mir::ScalarType::Ptr scalar_type) {
		switch (scalar_type->type) {
			case mir::ScalarType::Type::INT: 
				oss << "int";
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
				oss << "std::string";
				break;
			default:
				std::cerr << "Invalid scalar type. Exiting with errors\n";
				exit(-1);
		}
	}


	int CodeGenGunrock::genIncludeStmts(void) {
		oss << "#include <gunrock/gunrock.h>" << std::endl;
		oss << "#include <gunrock/graphio/graphio.cuh>" << std::endl;
		oss << "#include <gunrock/app/test_base.cuh>" << std::endl;
		oss << "#include <gunrock/app/app_base.cuh>" << std::endl;
		oss << "#include <gunrock/app/frontier.cuh>" << std::endl;
		oss << "#include <gunrock/app/enactor_types.cuh>" << std::endl;
		oss << "#include <gunrock/oprtr/oprtr.cuh>" << std::endl;

		oss << "using namespace gunrock;" << std::endl;

		oss << "typedef uint32_t VertexT;" << std::endl;
		oss << "typedef size_t SizeT;" << std::endl;
		oss << "typedef float ValueT;" << std::endl;

		oss << "typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_EDGE_VALUES | graph::HAS_CSR> WGraphT;" << std::endl; 
		oss << "typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR> GraphT;" << std::endl; 
		
		oss << "typedef typename app::Frontier<VertexT, SizeT, util::ARRAY_NONE, 0> FrontierT;" << std::endl;
		oss << "typedef typename oprtr::OprtrParameters<GraphT, Frontier, VertexT> OprtrParametersT;" << std::endl;
		oss << "typedef typename oprtr::OprtrParameters<WGraphT, Frontier, VertexT> WOprtrParametersT;" << std::endl;


		return 0;
	}



	void CodeGenGunrock::visit(mir::FuncDecl::Ptr func_decl) {
		oss << "// Generating code for function - " <<  func_decl->name << std::endl;

		// Special case for the main function

		if (func_decl->name == "main") {
			func_decl->isFunctor = false;
			oss << "int main(int argc, char *argv[]) {" << std::endl;
			indent();
			printIndent();
			oss << "util::SetDevice(3);" << std::endl;



			for (auto stmt : mir_context_->edgeset_alloc_stmts) {
				mir::AssignStmt::Ptr assign_stmt = std::dynamic_pointer_cast<mir::AssignStmt>(stmt);
				assert(assign_stmt != nullptr);
				mir::EdgeSetLoadExpr::Ptr edge_set_load_expr = std::dynamic_pointer_cast<mir::EdgeSetLoadExpr>(assign_stmt->expr);
				assert(edge_set_load_expr != nullptr);

				mir::VarExpr::Ptr lhs_var = std::dynamic_pointer_cast<mir::VarExpr>(assign_stmt->lhs);
				assert(lhs_var != nullptr);	

				std::string var_name = lhs_var->var.getName();

				printIndent();
				oss << "{" << std::endl;
				indent();

				printIndent();
				oss << "util::Parameters parameters(\"\");" << std::endl;
				printIndent();
				oss << "graphio::UseParameters(parameters);" << std::endl;
				printIndent();
				oss << "parameters.Set(\"graph-type\", \"market\")" << std::endl;
				printIndent();
				oss << "parameters.Set(\"graph-file\", ";
				edge_set_load_expr->file_name->accept(this);
				oss << ");" << std::endl;
				printIndent();
				oss << "graphio::LoadGraph(parameters, " << var_name << ");" << std::endl;

				dedent();
				printIndent();
				oss << "}" << std::endl;

			}


			for (auto constant : mir_context_->getLoweredConstants()) {
				if (auto type = std::dynamic_pointer_cast<mir::VectorType>(constant->type)){
					genPropertyArrayAlloca(constant);
				}
			}


			for (auto stmt: mir_context_->field_vector_init_stmts) {
				stmt->accept(this);
			}



		}else {
			if(func_decl->result.isInitialized()) {
				func_decl->result.getType()->accept(this);
				oss << " ";
			} else 
				oss << "void ";
			oss << func_decl->name << "(";
			bool print_delimeter = false;
			for (auto arg: func_decl->args) {
				if(print_delimeter)
					oss << ", ";
				arg.getType()->accept(this);
				oss << " " << arg.getName();

				print_delimeter = true;
			}
			oss <<") {" << std::endl;
			indent();
			if (func_decl->result.isInitialized()) {
				mir::VarDecl::Ptr var_decl = std::make_shared<mir::VarDecl>();
				var_decl->name = func_decl->result.getName();
				var_decl->type = func_decl->result.getType();
				var_decl->accept(this);
			}

		}


		assert(func_decl->body->stmts);

		func_decl->body->accept(this);	


		if (func_decl->name == "main") {	
			for (auto stmt : mir_context_->getLoweredConstants()) {
				printIndent();
				std::string var_name = stmt->name;	
				oss << var_name << ".Release();" << std::endl;
			}
			printIndent();
			oss << "return 0;" << std::endl;
			
		}

		if (func_decl->result.isInitialized()) {
			printIndent();
			oss << "return " << func_decl->result.getName() << std::endl;
		}


		dedent();
		printIndent();
		oss << "}" << std::endl;


		return;
	}

	void CodeGenGunrock::visit(mir::VarDecl::Ptr var_decl) {
		printIndent();
		var_decl->type->accept(this);
		oss << " " << var_decl->name;
		if (var_decl->initVal != nullptr) {
			oss << " = ";
			var_decl->initVal->accept(this);
		}
		oss << ";" << std::endl;
	}


	void CodeGenGunrock::visit(mir::StringLiteral::Ptr string_literal) {
		oss << "\"" << string_literal->val << "\"";		
	}
	void CodeGenGunrock::visit(mir::IntLiteral::Ptr int_literal) {
		oss << int_literal->val;
	}
	void CodeGenGunrock::visit(mir::BoolLiteral::Ptr bool_literal) {
		oss << bool_literal->val?"true":"false";
	}


	void CodeGenGunrock::visit(mir::TensorArrayReadExpr::Ptr tare) {
		tare->target->accept(this);
		oss << "[";
		tare->index->accept(this);
		oss << "]";	
	}

	void CodeGenGunrock::visit(mir::ExprStmt::Ptr expr_stmt) {
		printIndent();
		expr_stmt->expr->accept(this);
		oss << ";" << std::endl;
	}


	void CodeGenGunrock::visit(mir::ElementType::Ptr element_type) {
		oss << "VertexT";
	}

	void CodeGenGunrock::visit(mir::VertexSetType::Ptr vertex_set_type) {
		oss << "FrontierT";
	}

	void CodeGenGunrock::visit(mir::AssignStmt::Ptr assign_stmt) {
		printIndent();
		assign_stmt->lhs->accept(this);
		oss << " = ";
		assign_stmt->expr->accept(this);
		oss << ";" << std::endl;		
	}


	void CodeGenGunrock::visit(mir::AddExpr::Ptr add_expr) {
		oss << "(";
		add_expr->lhs->accept(this);
		oss << " + ";
		add_expr->rhs->accept(this);
		oss << ")";
		
	}

	void CodeGenGunrock::visit(mir::ReduceStmt::Ptr reduce_stmt) {
		switch (reduce_stmt->reduce_op_) {
			case mir::ReduceStmt::ReductionOp::MIN:
				printIndent();
				oss << "if ((";
				reduce_stmt->lhs->accept(this);
				oss << ") > (";
				reduce_stmt->expr->accept(this);
				oss << ")) { " << std::endl;
				indent();
				printIndent();
				reduce_stmt->lhs->accept(this);
				oss << " = ";
				reduce_stmt->expr->accept(this);
				oss << ";" << std::endl;
				if (reduce_stmt->tracking_var_name_ != "") {
					printIndent();
					oss << reduce_stmt->tracking_var_name_ << " = true ; " << std::endl;
				}
				dedent();
				printIndent();
				oss << "}" << std::endl;
				break;
			default:
				std::cerr << "Reduction operator not implemented yet.";
				break;	
		}
		
	}

	void CodeGenGunrock::visit(mir::VertexSetApplyExpr::Ptr vsae) {
		auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(vsae->target);
		if(mir_context_->isConstVertexSet(mir_var->var.getName())){
			auto associated_element_type = mir_context_->getElementTypeFromVectorOrSetName(mir_var->var.getName());
			assert(associated_element_type != nullptr);
			auto associated_element_type_size = mir_context_->getElementCount(associated_element_type);
			assert(associated_element_type != nullptr);
			if(!vsae->is_parallel) {
				std::cerr << "Non parallel VertexSetApplyExpr not supported yet. Exiting with failure" << std::endl;
				exit(-1);
			}
			oss << "{" << std::endl;
			indent();
			printIndent();
			oss << "auto apply_lambda = [] __device__ (VertexT *dummy, const SizeT &vertex_) {" << std::endl;
			indent();
			printIndent();
			oss << vsae->input_function_name << "(vertex_);" << std::endl;
			dedent();
			printIndent();
			oss << "}" << std::endl;

			printIndent();
			oss << "oprtr::ForAll((VertexT*) NULL, apply_lambda, ";
			associated_element_type_size->accept(this);
			oss << ", util::DEVICE, 0);" << std::endl;

			dedent();
			printIndent();
			oss << "}";				

		}else {
			std::cerr << "Dynamic set VertexSetApplyExpr not supported yet. Exiting with failure" << std::endl;
			exit(-1);
		}
	}

	void CodeGenGunrock::visit(mir::ForStmt::Ptr for_stmt) {
		printIndent();
		auto for_domain = for_stmt->domain;
		auto loop_var = for_stmt->loopVar;
		oss << "for (int " << loop_var << " = "; 
		for_domain->lower->accept(this);
		oss << "; " << loop_var << " < ";
		for_domain->upper->accept(this);
		oss << "; " << loop_var << "++ ) {" << std::endl;
		indent();
		for_stmt->body->accept(this);
		dedent();
		printIndent();
		oss << "}" << std::endl;
	}

	void CodeGenGunrock::visit(mir::WhileStmt::Ptr while_stmt) {
		printIndent();
		oss << "while (";
		while_stmt->cond->accept(this);
		oss << ") {" << std::endl;
		indent();
		while_stmt->body->accept(this);
		dedent();
		printIndent();
		oss << "}" << std::endl;	
	}
	
	void CodeGenGunrock::visit(mir::IfStmt::Ptr if_stmt) {
		printIndent();
		oss << "if (";
		if_stmt->cond->accept(this);
		oss << ") {" << std::endl;
		indent();
		if_stmt->ifBody->accept(this);
		dedent();
		if(if_stmt->elseBody) {
			printIndent();
			oss << "} else { " << std::endl;
			indent();
			if_stmt->elseBody->accept(this);
			dedent();	
		}
		printIndent();
		oss << "}" << std::endl;	
	}

	void CodeGenGunrock::visit(mir::BreakStmt::Ptr break_stmt) {
		printIndent();
		oss << "break;" << std::endl;
	}

	void CodeGenGunrock::visit(mir::EqExpr::Ptr eq_expr) {
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
					std::cerr << "Invalid operator for EqExpr" << std::endl;

			}
			oss << "(";
			eq_expr->operands[i+1]->accept(this);
			oss << ")";
		}
	}

	void CodeGenGunrock::visit(mir::PrintStmt::Ptr print_stmt) {
		printIndent();
		oss << "std::cout << ";
		print_stmt->expr->accept(this);
		oss << ";" << std::endl;
	}
	int CodeGenGunrock::genEdgeSets(void) {
		for (auto edgeset : mir_context_->getEdgeSets()) {
			auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
			if (edge_set_type->weight_type != nullptr) {
				oss << "WGraphT " << edgeset->name << ";" << std::endl;
			} else {
				oss << "GraphT " << edgeset->name << ";" << std::endl;
			}
		}
		return 0;
	}
}
