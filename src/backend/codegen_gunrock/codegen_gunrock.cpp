#include "graphit/backend/codegen_gunrock/codegen_gunrock.h"
#include "graphit/backend/codegen_gunrock/assign_function_context.h"
#include <cctype>
#include <cstdlib>
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

		AssignFunctionContext assign_function_context(mir_context_);
		assign_function_context.assign_function_context();
				
		genIncludeStmts();
		genEdgeSets();

		genVertexSets();

		for (auto constant : mir_context_->getLoweredConstants()) {
			if (auto type = std::dynamic_pointer_cast<mir::VectorType>(constant->type)){
				genPropertyArrayDecl(constant);
			}else{
				constant->type->accept(this);
				oss << " " << constant->name << ";" << std::endl;	
			}
		}

		// Process all functions 	
		const std::vector<mir::FuncDecl::Ptr> &functions = mir_context_->getFunctionList();	

		for (auto it = functions.begin(); it != functions.end(); it++) {
			it->get()->accept(this);
		}
		return 0;
	}

	int CodeGenGunrock::fillLambdaBody(mir::FuncDecl::Ptr func_decl, std::vector<std::string> args) {
		int counter = 0;
		for(auto arg:func_decl->args){
			printIndent();
			oss << "auto &" << arg.getName() << " = " << args[counter++] <<";"<< std::endl;
		}	
		auto old_context = current_context;
		current_context = mir::FuncDecl::CONTEXT_DEVICE;
		
		if (func_decl->result.isInitialized()) {
			mir::VarDecl::Ptr var_decl = std::make_shared<mir::VarDecl>();
			var_decl->name = func_decl->result.getName();
			var_decl->type = func_decl->result.getType();
			var_decl->accept(this);
		}
		func_decl->body->accept(this);	
		if (func_decl->result.isInitialized()) {
			printIndent();
			oss << "return " << func_decl->result.getName() << ";" << std::endl;
		}
		current_context = old_context;
	}

	int CodeGenGunrock::genVertexSets(void) {
		for (auto var:mir_context_->getConstVertexSets()){
			oss << "FrontierT " << var->name << ";" << std::endl;
			 
			mir::EdgeSetType::Ptr edge_set = mir_context_->getEdgeSetTypeFromElementType(mir::to<mir::VertexSetType>(var->type)->element);
			if (edge_set->weight_type == nullptr)
				oss << "OprtrParametersT " << var->name << "_parameters;" << std::endl;
			else 
				oss << "WOprtrParametersT " << var->name << "_parameters;" << std::endl;
		}
	}

	int CodeGenGunrock::genPropertyArrayDecl(mir::VarDecl::Ptr var_decl) {
		//oss << "// " << var_decl->name << std::endl;

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
	std::string CodeGenGunrock::getAllGlobals(void) {
		std::string output;
		for (auto edgeset : mir_context_->getEdgeSets()) {
			output += (edgeset->name + ", ");
		}
		for (auto constant : mir_context_->getLoweredConstants()) {
			output += (constant->name + ", ");
		}
		return output.substr(0, output.length()-2);
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
		/*
		oss << "#include <gunrock/gunrock.h>" << std::endl;
		oss << "#include <gunrock/graphio/graphio.cuh>" << std::endl;
		oss << "#include <gunrock/app/test_base.cuh>" << std::endl;
		oss << "#include <gunrock/app/app_base.cuh>" << std::endl;
		oss << "#include <gunrock/app/frontier.cuh>" << std::endl;
		oss << "#include <gunrock/app/enactor_types.cuh>" << std::endl;
		oss << "#include <gunrock/oprtr/oprtr.cuh>" << std::endl;
		*/


		oss << "#include \"builtins_gunrock.h\"" << std::endl;

		oss << "using namespace gunrock;" << std::endl;

		oss << "typedef uint32_t VertexT;" << std::endl;
		oss << "typedef size_t SizeT;" << std::endl;
		oss << "typedef int ValueT;" << std::endl;

		oss << "typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_EDGE_VALUES | graph::HAS_CSR> WGraphT;" << std::endl; 
		oss << "typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR> GraphT;" << std::endl; 
		
		oss << "typedef typename app::Frontier<VertexT, SizeT, util::ARRAY_NONE, 0> FrontierT;" << std::endl;
		oss << "typedef typename oprtr::OprtrParameters<GraphT, FrontierT, VertexT> OprtrParametersT;" << std::endl;
		oss << "typedef typename oprtr::OprtrParameters<WGraphT, FrontierT, VertexT> WOprtrParametersT;" << std::endl;


		return 0;
	}



	void CodeGenGunrock::visit(mir::FuncDecl::Ptr func_decl) {
		//oss << "// Generating code for function - " <<  func_decl->name << std::endl;

		// Special case for the main function

		// Do not generate code for Device functions for now, the body will be generated as a lambda
		if (func_decl->realized_context & mir::FuncDecl::CONTEXT_DEVICE)
			return; 

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
				oss << "parameters.Set(\"graph-type\", \"market\");" << std::endl;
				//printIndent();
				//oss << "parameters.Set(\"undirected\", false);" << std::endl;
				printIndent();
				oss << "parameters.Set(\"graph-file\", ";
				edge_set_load_expr->file_name->accept(this);
				oss << ");" << std::endl;
				printIndent();
				oss << "graphio::LoadGraph(parameters, " << var_name << ");" << std::endl;

				dedent();
				printIndent();
				oss << "}" << std::endl;
				printIndent();
				oss << var_name << ".Move(util::HOST, util::DEVICE, 0);" << std::endl;

			}


		
				
			for (auto var:mir_context_->getConstVertexSets()){
				mir::VertexSetAllocExpr::Ptr vsae = std::make_shared<mir::VertexSetAllocExpr>();
				vsae->vertex_set_name = var->name;
				vsae->element_type = mir::to<mir::VertexSetType>(var->type)->element;
				vsae->accept(this);
				printIndent();
				oss << "builtin_addAllVertices(" << var->name << ", ";
				mir_context_->getElementCount(vsae->element_type)->accept(this);
				oss << ");" << std::endl;
				
			}


			for (auto constant : mir_context_->getLoweredConstants()) {
				if (auto type = std::dynamic_pointer_cast<mir::VectorType>(constant->type)){
					if (constant->needs_allocation)
						genPropertyArrayAlloca(constant);
				}else{
					if (constant->initVal != nullptr){
						printIndent();
						oss << constant->name  << " = ";
						constant->initVal->accept(this);
						oss << ";" << std::endl;
					}	
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
			std::string target_config = (func_decl->realized_context & mir::FuncDecl::CONTEXT_DEVICE)?"__device__":"__host__";
			oss << func_decl->name << " " << target_config << " (";
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

		current_context = func_decl->realized_context;

		func_decl->body->accept(this);	


		if (func_decl->name == "main") {	
			for (auto stmt : mir_context_->getLoweredConstants()) {
				if (auto type = mir::to<mir::VectorType>(stmt->type) && stmt->needs_allocation) {
					printIndent();
					std::string var_name = stmt->name;	
					oss << var_name << ".Release();" << std::endl;
				}
			}
			printIndent();
			oss << "return 0;" << std::endl;	
		}

		if (func_decl->result.isInitialized()) {
			printIndent();
			oss << "return " << func_decl->result.getName() << ";" << std::endl;
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

	
		if (var_decl->initVal != nullptr && !mir::isa<mir::VertexSetType>(var_decl->type)) {
			oss << " = ";
			var_decl->initVal->accept(this);
		}
		
		oss << ";" << std::endl;
		if (var_decl->initVal != nullptr && mir::isa<mir::VertexSetType>(var_decl->type)) {
			mir::EdgeSetType::Ptr edge_set = mir_context_->getEdgeSetTypeFromElementType(mir::to<mir::VertexSetType>(var_decl->type)->element);
			printIndent();
			if (edge_set->weight_type == nullptr)
				oss << "OprtrParametersT " << var_decl->name << "_parameters;" << std::endl;
			else 
				oss << "WOprtrParametersT " << var_decl->name << "_parameters;" << std::endl;
			mir::to<mir::VertexSetAllocExpr>(var_decl->initVal) -> vertex_set_name = var_decl->name;
			var_decl->initVal->accept(this);
		}
	}


	void CodeGenGunrock::visit(mir::VertexSetAllocExpr::Ptr vsae) {
		printIndent();
		oss << vsae->vertex_set_name << ".Init(2, NULL, \"\", util::DEVICE | util::HOST);" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << ".Reset();" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << ".queue_index = 0;" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << ".queue_length = 0;" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << ".queue_reset = true;" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << ".work_progress.Reset_();" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << "_parameters.Init();" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << "_parameters.stream = 0;" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << "_parameters.frontier = &" << vsae->vertex_set_name << ";" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << "_parameters.cuda_props = NULL;" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << "_parameters.advance_mode = \"LB_CULL\";" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << "_parameters.filter_mode = \"CULL\";" << std::endl;
		printIndent();
		oss << vsae->vertex_set_name << "_parameters.context = mgpu::CreateCudaDeviceAttachStream(3, 0);" << std::endl;
		printIndent();
		oss << "{" << std::endl;
		indent();
		printIndent();
		oss << "std::vector<double> queue_factors;" << std::endl;
		printIndent();
		oss << "queue_factors.push_back(6.0);" << std::endl;
		printIndent();
		oss << "queue_factors.push_back(6.0);" << std::endl;
		printIndent();
		
		mir::Expr::Ptr size_expr = mir_context_->getElementCount(vsae->element_type);
		
		oss << vsae->vertex_set_name << ".Allocate(";
		size_expr->accept(this);

		mir::Expr::Ptr arg = mir::to<mir::Call>(size_expr)->args[0];

		oss <<", builtin_getEdges(" << mir::to<mir::VarExpr>(arg)->var.getName() <<  "), queue_factors);" << std::endl;



		dedent();
		printIndent();
		oss << "}" << std::endl; 
			
		
	}


	void CodeGenGunrock::visit(mir::StmtBlock::Ptr stmt_block) {
		for (auto stmt: *(stmt_block->stmts)) {
			if(current_context == mir::FuncDecl::CONTEXT_DEVICE)
				stmt->accept(this);
			else {
				ExtractReadWriteSet extractor;
				stmt->accept(&extractor);

				for (auto var: extractor.getReadSet()) {
					auto vector_type = mir::to<mir::VectorType>(var.getType());
					auto size_expr = mir_context_->getElementCount(vector_type->element_type);
					std::string name = var.getName();
					printIndent();
					oss << name << ".Move(util::DEVICE, util::HOST, ";
					size_expr->accept(this);
					oss << ", 0, 0);" << std::endl;						
				}
				stmt->accept(this);
				for (auto var: extractor.getWriteSet()) {
					auto vector_type = mir::to<mir::VectorType>(var.getType());
					auto size_expr = mir_context_->getElementCount(vector_type->element_type);
					std::string name = var.getName();
					printIndent();
					oss << name << ".Move(util::HOST, util::DEVICE, ";
					size_expr->accept(this);
					oss << ", 0, 0);" << std::endl;						
				}	
			}
		}
		
	}

	void CodeGenGunrock::visit(mir::VertexSetWhereExpr::Ptr vswe) {
		if (vswe->input_func == "") {
			std::cerr << "VertexSetWhereExpr without a from function. Exiting with failure.\n";
			exit(-1);
		} 
		if (vswe->var == nullptr) {
			std::cerr << "VertexSetWhereExpr result not assigned. Exiting with failure\n";
			exit(-1);
		}
		auto mir_var = mir_context_->getSymbol(vswe->target);
		oss << "{" << std::endl;
		indent();
		for(auto var:mir_context_->getLoweredConstants()){
			printIndent();
			oss << "auto " << var->name << " = ::" << var->name << ";" << std::endl;
		}
		for (auto edgeset : mir_context_->getEdgeSets()) {
			printIndent();
			oss << "auto " << edgeset->name << " = ::" << edgeset->name << ";" << std::endl;
		}
		printIndent();
		oss << "auto apply_lambda = [" << getAllGlobals() << "] __device__ (const VertexT &_src, VertexT &_dest, const SizeT &_edge_id, const VertexT &_input_item, const SizeT &_input_pos, SizeT &_output_pos) -> bool {" << std::endl;
		indent();
		//printIndent();
		//oss << "return " << vsae->input_function_name << "(dest);" << std::endl;	
		fillLambdaBody(mir_context_->getFunction(vswe->input_func), {"_dest"});
		dedent();
		printIndent();
		oss << "};" << std::endl;
		mir::VertexSetType::Ptr vst = mir::to<mir::VertexSetType>(mir_var.getType());
		mir::ElementType::Ptr element_type = vst->element;


		if (vswe->var->getName() != mir_var.getName()){
			printIndent();
			oss << mir_var.getName() << ".GetQueueLength(0);" << std::endl;
			printIndent();
			oss << "auto old_length = " << mir_var.getName() << ".queue_length;" << std::endl;
		}
		printIndent();
		std::string graph_name = mir_context_->getEdgeSetNameFromEdgeSetType(mir_context_->getEdgeSetTypeFromElementType(element_type));
		assert(graph_name != "");

		oss << "oprtr::Filter<oprtr::OprtrType_V2V>(" << graph_name << ".csr(), " << mir_var.getName() << ".V_Q(), " << mir_var.getName() << ".Next_V_Q(), " << mir_var.getName() << "_parameters, apply_lambda);" << std::endl;
		printIndent();
		oss << "cudaDeviceSynchronize();" << std::endl;


		if (vswe->var->getName() != mir_var.getName()) {

			/*
			auto tmp_queue = vswe->var->getName().V_Q();
			vswe->var->getName(). queue_map[queue_index % 2] = mir_var.getName().V_Q();
			mir_var.getName().queue_map[queue_index %2] = tmp_queue;
			mir_var.queue_index = ((mir_var.queue_index-1) % 2);
			mir_var.work_progress.SetQueueLength(old_length);
			*/

			printIndent();
			oss << mir_var.getName() << ".GetQueueLength(0);" << std::endl;

			printIndent();
			oss << "auto tmp_queue = *(" << vswe->var->getName() << ".V_Q());" << std::endl;
			printIndent();
			oss << "*(" << vswe->var->getName() << ".V_Q()) = *(" << mir_var.getName() << ".V_Q());" << std::endl;
			printIndent();
			oss << "*(" << mir_var.getName() << ".V_Q()) = tmp_queue;" << std::endl;
			printIndent();
			oss << mir_var.getName() << ".queue_index = ((" << mir_var.getName() << ".queue_index - 1) % 2);" << std::endl;
			printIndent();	
			oss << vswe->var->getName() << ".queue_length = " << mir_var.getName() << ".queue_length;" << std::endl;
			printIndent();
			oss << vswe->var->getName() << ".work_progress.SetQueueLength("<< vswe->var->getName() << ".queue_index, " << vswe->var->getName() << ".queue_length);" << std::endl;

			printIndent();
			oss << mir_var.getName() << ".queue_length = old_length;" << std::endl;
			printIndent();
			oss << mir_var.getName() << ".work_progress.SetQueueLength(" << mir_var.getName() << ".queue_index, old_length);" << std::endl;
			printIndent();
			oss << mir_var.getName() << ".work_progress.SetQueueLength((" << mir_var.getName() << ".queue_index+1)%2, 0);" << std::endl;
			

			//vswe->var->getName();
		}


	
		vswe->var->getName();
		dedent();
		printIndent();
		oss << "}";
	}

	void CodeGenGunrock::visit(mir::StringLiteral::Ptr string_literal) {
		oss << "\"";
		for (auto ch : string_literal->val)
			if (iscntrl(ch) || ch == '\\' || ch == '\"' || ch == '\'') {
				oss << "\\0" << std::oct << (int)(ch);	
			}else
				oss << ch;
		oss << "\"";		
	}
	void CodeGenGunrock::visit(mir::IntLiteral::Ptr int_literal) {
		oss << int_literal->val;
	}
	void CodeGenGunrock::visit(mir::FloatLiteral::Ptr float_literal) {
		oss << "((float)" << float_literal->val << ")";
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
		if (mir::isa<mir::VertexSetApplyExpr>(assign_stmt->expr)){
			mir::to<mir::VertexSetApplyExpr>(assign_stmt->expr)->var = &(mir::to<mir::VarExpr>(assign_stmt->lhs)->var);
			assign_stmt->expr->accept(this);
			oss << ";" << std::endl;	
			
		}else if(mir::isa<mir::PushEdgeSetApplyExpr>(assign_stmt->expr)){
			mir::to<mir::PushEdgeSetApplyExpr>(assign_stmt->expr)->var = &(mir::to<mir::VarExpr>(assign_stmt->lhs)->var);
			assign_stmt->expr->accept(this);
			oss << ";" << std::endl;
		}else if(mir::isa<mir::VertexSetWhereExpr>(assign_stmt->expr)){
			mir::to<mir::VertexSetWhereExpr>(assign_stmt->expr)->var = &(mir::to<mir::VarExpr>(assign_stmt->lhs)->var);
			assign_stmt->expr->accept(this);
			oss << ";" << std::endl;
		}else{
			assign_stmt->lhs->accept(this);
			oss << " = ";
			assign_stmt->expr->accept(this);
			oss << ";" << std::endl;		
		}
	}


	void CodeGenGunrock::visit(mir::AddExpr::Ptr add_expr) {
		oss << "(";
		add_expr->lhs->accept(this);
		oss << " + ";
		add_expr->rhs->accept(this);
		oss << ")";
		
	}
	void CodeGenGunrock::visit(mir::SubExpr::Ptr sub_expr) {
		oss << "(";
		sub_expr->lhs->accept(this);
		oss << " - ";
		sub_expr->rhs->accept(this);
		oss << ")";
		
	}

	void CodeGenGunrock::visit(mir::MulExpr::Ptr mul_expr) {
		oss << "(";
		mul_expr->lhs->accept(this);
		oss << " * ";
		mul_expr->rhs->accept(this);
		oss << ")";
		
	}
	void CodeGenGunrock::visit(mir::DivExpr::Ptr div_expr) {
		oss << "(";
		div_expr->lhs->accept(this);
		oss << " / ";
		div_expr->rhs->accept(this);
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

			case mir::ReduceStmt::ReductionOp::ATOMIC_MIN:
				printIndent();
				if (reduce_stmt->tracking_var_name_ != "")
					oss << reduce_stmt->tracking_var_name_ << " = ";
				oss << "builtin_writeMin((";
				reduce_stmt->lhs->accept(this);
				oss << "), ";
				reduce_stmt->expr->accept(this);
				oss << ");" << std::endl;
				break;
			case mir::ReduceStmt::ReductionOp::ATOMIC_SUM:
				printIndent();
				if(reduce_stmt->tracking_var_name_ != "")
					oss << reduce_stmt->tracking_var_name_ << " = ";
				oss << "builtin_writeSum((";
				reduce_stmt->lhs->accept(this);
				oss << "), ";
				reduce_stmt->expr->accept(this);
				oss << ");" << std::endl;
				break;
			case mir::ReduceStmt::ReductionOp::SUM:
				printIndent();
				reduce_stmt->lhs->accept(this);
				oss << " += ";
				reduce_stmt->expr->accept(this);
				oss << ";" << std::endl;
				if (reduce_stmt->tracking_var_name_ != "") {
					printIndent();
					oss << reduce_stmt->tracking_var_name_ << " = (";
					reduce_stmt->expr->accept(this);
					oss << ") == 0;" << std::endl;
				}
				break;
			default:
				std::cerr << "Reduction operator not implemented yet." << std::endl;
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
			
			for(auto var:mir_context_->getLoweredConstants()){
				printIndent();
				oss << "auto " << var->name << " = ::" << var->name << ";" << std::endl;
			}
			for (auto edgeset : mir_context_->getEdgeSets()) {
				printIndent();
				oss << "auto " << edgeset->name << " = ::" << edgeset->name << ";" << std::endl;
			}
				
			printIndent();
			oss << "auto apply_lambda = [" << getAllGlobals() << "] __device__ (VertexT *_dummy, const SizeT &_vertex) {" << std::endl;
			indent();
			//printIndent();
			//oss << vsae->input_function_name << "(vertex_);" << std::endl;
			fillLambdaBody(mir_context_->getFunction(vsae->input_function_name), {"_vertex"});
			dedent();
			printIndent();
			oss << "};" << std::endl;

			printIndent();
			oss << "oprtr::ForAll((VertexT*) NULL, apply_lambda, ";
			associated_element_type_size->accept(this);
			oss << ", util::DEVICE, 0);" << std::endl;
			printIndent();
			oss << "cudaDeviceSynchronize();" << std::endl;
			dedent();
			printIndent();
			oss << "}";				

		}else {	
			auto mir_var = mir::to<mir::VarExpr>(vsae->target);
			oss << "{" << std::endl;
			indent();
			for(auto var:mir_context_->getLoweredConstants()){
				printIndent();
				oss << "auto " << var->name << " = ::" << var->name << ";" << std::endl;
			}
			for (auto edgeset : mir_context_->getEdgeSets()) {
				printIndent();
				oss << "auto " << edgeset->name << " = ::" << edgeset->name << ";" << std::endl;
			}
			printIndent();
			oss << "auto apply_lambda = [" << getAllGlobals() << "] __device__ (const VertexT &_src, VertexT &_dest, const SizeT &_edge_id, const VertexT &_input_item, const SizeT &_input_pos, SizeT &_output_pos) -> bool {" << std::endl;
			indent();
			//printIndent();
			//oss << "return " << vsae->input_function_name << "(dest);" << std::endl;	
			fillLambdaBody(mir_context_->getFunction(vsae->input_function_name), {"_dest"});
			dedent();
			printIndent();
			oss << "};" << std::endl;
			mir::VertexSetType::Ptr vst = mir::to<mir::VertexSetType>(mir_var->var.getType());
			mir::ElementType::Ptr element_type = vst->element;
			printIndent();
			std::string graph_name = mir_context_->getEdgeSetNameFromEdgeSetType(mir_context_->getEdgeSetTypeFromElementType(element_type));
			assert(graph_name != "");
			if (vsae->var == nullptr)
				oss << "oprtr::Filter<oprtr::OprtrType_V2V>(" << graph_name << ".csr(), " << mir_var->var.getName() << ".V_Q(), nullptr, " << mir_var->var.getName() << "_parameters, apply_lambda);" << std::endl;
			else
				oss << "oprtr::Filter<oprtr::OprtrType_V2V(" << graph_name << ".csr(), " << mir_var->var.getName() << ".V_Q(), " << vsae->var->getName() << ".Next_V_Q(), " << mir_var->var.getName() << "_parameters, apply_lambda);" << std::endl;
			printIndent();
			oss << "cudaDeviceSynchronize();" << std::endl;
			dedent();
			printIndent();
			oss << "}";
		}
	}
	void CodeGenGunrock::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
		auto mir_var = mir::to<mir::VarExpr>(pesae->target);
		if(!mir_context_->isEdgeSet(mir_var->var.getName())) {
			std::cerr << "Edge set apply on non const edge sets not supporeted yet. Exiting with failure" << std::endl;
			exit(-1);
		}

		if(mir_context_->isFunction(pesae->from_func)){
			std::cerr << "Edge set apply from funcyion not supported yet. Exiting with failure" << std::endl;
			exit(-1);
		}

		if(pesae->from_func=="") {
			if (pesae->var != nullptr) {
				std::cerr << "Output frontier without input frontier not supported yet. Exiting with failure" << std::endl;
				std::cerr << pesae->var->getName() << std::endl;
				exit(-1);
			}
			oss << "{" << std::endl;
			indent();
			for(auto var:mir_context_->getLoweredConstants()){
				printIndent();
				oss << "auto " << var->name << " = ::" << var->name << ";" << std::endl;
			}
			for (auto edgeset : mir_context_->getEdgeSets()) {
				printIndent();
				oss << "auto " << edgeset->name << " = ::" << edgeset->name << ";" << std::endl;
			}
			printIndent();
			oss << "auto apply_lambda = [" << getAllGlobals() << "] __device__ (const VertexT *_dummy, const SizeT &_e) -> bool {" << std::endl;
			indent();
			printIndent();
			oss << "VertexT _src, _dest;" << std::endl; 
			printIndent();
			oss << mir_var->var.getName() << ".GetEdgeSrcDest(_e, _src, _dest);" << std::endl;
			auto edge_set_type = mir::to<mir::EdgeSetType>(mir_context_->getConstEdgeSetByName(mir_var->var.getName())->type);
			if (edge_set_type->weight_type != nullptr){
				printIndent();
				oss << "auto _edge_weight = Load<cub::LOAD_CS>(" << mir_var->var.getName() << ".CsrT::edge_values + _e);" << std::endl;
			}
			fillLambdaBody(mir_context_->getFunction(pesae->input_function_name), {"_src", "_dest", "_edge_weight"});
				
			dedent();
			printIndent();
			oss << "};" << std::endl;
			printIndent();
			oss << "oprtr::ForAll((VertexT*) NULL, apply_lambda, " << mir_var->var.getName() << ".edges, util::DEVICE, 0);" << std::endl;
			printIndent();
			oss << "cudaDeviceSynchronize();" << std::endl;
			dedent();
			printIndent();
			oss << "}";
			
		} else {
			oss << "{" << std::endl;
			indent();
			for(auto var:mir_context_->getLoweredConstants()){
				printIndent();
				oss << "auto " << var->name << " = ::" << var->name << ";" << std::endl;
			}
			for (auto edgeset : mir_context_->getEdgeSets()) {
				printIndent();
				oss << "auto " << edgeset->name << " = ::" << edgeset->name << ";" << std::endl;
			}
			printIndent();
			oss << "auto apply_lambda = [" << getAllGlobals() << "] __device__ (const VertexT &_src, VertexT &_dest, const SizeT &_edge_id, const VertexT &_input_item, const SizeT &_input_pos, SizeT &_output_pos) -> bool {" << std::endl;
			indent();
			
			auto edge_set_type = mir::to<mir::EdgeSetType>(mir_context_->getConstEdgeSetByName(mir_var->var.getName())->type);
			if (edge_set_type->weight_type != nullptr){
				printIndent();
				oss << "auto _edge_weight = Load<cub::LOAD_CS>(" << mir_var->var.getName() << ".CsrT::edge_values + _edge_id);" << std::endl;
			}
			fillLambdaBody(mir_context_->getFunction(pesae->input_function_name), {"_src", "_dest", "_edge_weight"});
			dedent();
			printIndent();
			oss << "};" << std::endl;
			printIndent();
			oss << "auto pass_lambda = [] __device__ (const VertexT &src, VertexT &dest, const SizeT &edge_id, const VertexT &input_item, const SizeT &input_pos, SizeT &output_pos) -> bool {" << std::endl;
			indent();
			printIndent();
			oss << "return true;" << std::endl;	
			dedent();
			printIndent();
			oss << "};" << std::endl;

			printIndent();	
			if(pesae->var == nullptr)
				oss << "oprtr::Advance<oprtr::OprtrType_V2V>(" << mir_var->var.getName() << ".csr(), " << pesae->from_func << ".V_Q(), nullptr, " << pesae->from_func << "_parameters, apply_lambda, pass_lambda);" << std::endl;
			else 
				oss << "oprtr::Advance<oprtr::OprtrType_V2V>(" << mir_var->var.getName() << ".csr(), " << pesae->from_func << ".V_Q(), " << pesae->var->getName() << ".Next_V_Q(), " << pesae->from_func << "_parameters, apply_lambda, pass_lambda);" << std::endl;
			printIndent();
			oss << "cudaDeviceSynchronize();" << std::endl;

			
			dedent();
			printIndent();
			oss << "}";	
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
		ExtractReadWriteSet extractor;
		for_stmt->accept(&extractor);
			
		for (auto var: extractor.getWriteSet()) {
			auto vector_type = mir::to<mir::VectorType>(var.getType());
			auto size_expr = mir_context_->getElementCount(vector_type->element_type);
			std::string name = var.getName();
			printIndent();
			oss << name << ".Move(util::HOST, util::DEVICE, ";
			size_expr->accept(this);
			oss << ", 0, 0);" << std::endl;						
		}	
		for_stmt->body->accept(this);
		for (auto var: extractor.getReadSet()) {
			auto vector_type = mir::to<mir::VectorType>(var.getType());
			auto size_expr = mir_context_->getElementCount(vector_type->element_type);
			std::string name = var.getName();
			printIndent();
			oss << name << ".Move(util::DEVICE, util::HOST, ";
			size_expr->accept(this);
			oss << ", 0, 0);" << std::endl;						
		}	
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
		
		ExtractReadWriteSet extractor;
		while_stmt->cond->accept(&extractor);
			
		for (auto var: extractor.getWriteSet()) {
			auto vector_type = mir::to<mir::VectorType>(var.getType());
			auto size_expr = mir_context_->getElementCount(vector_type->element_type);
			std::string name = var.getName();
			printIndent();
			oss << name << ".Move(util::HOST, util::DEVICE, ";
			size_expr->accept(this);
			oss << ", 0, 0);" << std::endl;						
		}	
		
		while_stmt->body->accept(this);

		for (auto var: extractor.getReadSet()) {
			auto vector_type = mir::to<mir::VectorType>(var.getType());
			auto size_expr = mir_context_->getElementCount(vector_type->element_type);
			std::string name = var.getName();
			printIndent();
			oss << name << ".Move(util::DEVICE, util::HOST, ";
			size_expr->accept(this);
			oss << ", 0, 0);" << std::endl;						
		}	

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

	void ExtractReadWriteSet::add_read(mir::Var var) {
		read_set_.push_back(var);
	}
	void ExtractReadWriteSet::add_write(mir::Var var) {
		write_set_.push_back(var);
	}
	void ExtractReadWriteSet::visit(mir::TensorArrayReadExpr::Ptr tare) {
		mir::Var target = mir::to<mir::VarExpr>(tare->target)->var;
		add_read(target);
		tare->index->accept(this);
	}
	void ExtractReadWriteSet::visit(mir::StmtBlock::Ptr stmt_block) {
		return;
	}
	void ExtractReadWriteSet::visit(mir::AssignStmt::Ptr assign_expr) {
		if (mir::isa<mir::TensorArrayReadExpr>(assign_expr->lhs)) {
			mir::Var target = mir::to<mir::VarExpr>(mir::to<mir::TensorArrayReadExpr>(assign_expr->lhs)->target)->var;
			add_write(target);
			mir::to<mir::TensorArrayReadExpr>(assign_expr->lhs)->index->accept(this);
			assign_expr->expr->accept(this);
		}else{
			assign_expr->lhs->accept(this);
			assign_expr->expr->accept(this);
		}
	}
	std::vector<mir::Var> ExtractReadWriteSet::getReadSet(void) {
		std::vector<std::string> read_inserted;
		std::vector<mir::Var> read_inserted_var;
		for (auto var: read_set_) {
			std::string name = var.getName();
			if (std::find(read_inserted.begin(), read_inserted.end(), name) == read_inserted.end()){
				read_inserted.push_back(name);
				read_inserted_var.push_back(var);
			}
		}
		for (auto var: write_set_) {
			std::string name = var.getName();
			if (std::find(read_inserted.begin(), read_inserted.end(), name) == read_inserted.end()) {
				read_inserted.push_back(name);
				read_inserted_var.push_back(var);
			}
		}	
		return read_inserted_var;
	}
	std::vector<mir::Var> ExtractReadWriteSet::getWriteSet(void) {
		std::vector<std::string> write_inserted;
		std::vector<mir::Var> write_inserted_var;
		for (auto var: write_set_) {
			std::string name = var.getName();
			if(std::find(write_inserted.begin(), write_inserted.end(), name) == write_inserted.end()) {
				write_inserted.push_back(name);
				write_inserted_var.push_back(var);
			}
		}
		return write_inserted_var;
	}
}
