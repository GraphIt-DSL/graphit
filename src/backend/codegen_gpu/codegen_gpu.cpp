//
// Created by Ajay Brahmakshatriya on 9/7/2019
//

#include <graphit/backend/codegen_gpu/codegen_gpu.h>
#include <graphit/backend/codegen_gpu/assign_function_context.h>
#include "graphit/backend/codegen_gpu/extract_read_write_set.h"
#include <graphit/midend/mir.h>
#include <cstring>
#include <iostream>

namespace graphit {
int CodeGenGPU::genGPU() {
	AssignFunctionContext assign_function_context(mir_context_);
	assign_function_context.assign_function_context();


	CodeGenGPUHost code_gen_gpu_host(oss, mir_context_, module_name, "");

	genIncludeStmts();
	
	genGlobalDeclarations();

	// This generates all the declarations of type GraphT<...>
	genEdgeSets();

	// Declare all the vertex properties
	// We are only declaring the device versions now. If required we can generate the host versions later
	for (auto constant: mir_context_->getLoweredConstants()) {
		if ((mir::isa<mir::VectorType>(constant->type))) {
			// This is some vertex data
			genPropertyArrayDecl(constant);	
		} else {
			// This is some scalar variable w or w/o initialization
			genScalarDecl(constant);
		}
	}	
		
	std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
	// Before we generate any functions or kernels, we generate the function declarations
	for (auto function: functions) {
		if (function->name != "main")
			genFuncDecl(function);
	}
	
	// Every operator requires a kernel to be generated
	// Create that first because all the actual functions will be calling these kernels
	CodeGenGPUKernelEmitter kernel_emitter(oss, mir_context_);
	for (auto function: functions)
		function->accept(&kernel_emitter);		
	
	// All the fused kernels need to generated before we can acutally generate the functions
	for (auto while_loop: mir_context_->fused_while_loops) 
		genFusedWhileLoop(while_loop);

	for (auto function: functions) {
		if (function->function_context & mir::FuncDecl::function_context_type::CONTEXT_DEVICE)
			function->accept(this);
		if (function->function_context & mir::FuncDecl::function_context_type::CONTEXT_HOST)
			function->accept(&code_gen_gpu_host);
	}

	oss << std::endl;
	return 0;
}

void CodeGenGPU::genScalarDecl(mir::VarDecl::Ptr var_decl) {	
	var_decl->type->accept(this);
	oss << " __device__ " << var_decl->name << "; " << std::endl;
	
	var_decl->type->accept(this);
	oss << " __host_" << var_decl->name << ";" << std::endl;

	if (mir::isa<mir::PriorityQueueType>(var_decl->type)) {
		var_decl->type->accept(this);
		oss << " *__device_" << var_decl->name << ";" << std::endl;
	}
}
void CodeGenGPU::genPropertyArrayDecl(mir::VarDecl::Ptr constant) {
	mir::VectorType::Ptr vector_type = mir::to<mir::VectorType>(constant->type);
	vector_type->vector_element_type->accept(this);
	oss << " __device__ *" << constant->name << ";" << std::endl;

	// Also generate the host versions of these arrays 
	vector_type->vector_element_type->accept(this);
	oss << " " << "*__host_" << constant->name << ";" << std::endl;
	// Also generate the device pointer for easy copy
	vector_type->vector_element_type->accept(this);
	oss << " " << "*__device_" << constant->name << ";" << std::endl;
}

void CodeGenGPU::genPropertyArrayAlloca(mir::VarDecl::Ptr var_decl) {
	auto vector_type = mir::to<mir::VectorType>(var_decl->type);
	assert(vector_type != nullptr);

	mir::Expr::Ptr size_expr = nullptr;	
	if (vector_type->element_type != nullptr) {
		size_expr = mir_context_->getElementCount(vector_type->element_type);
		assert(size_expr != nullptr);
	}
	
	
	if (var_decl->initVal != nullptr && mir::isa<mir::Call>(var_decl->initVal)) {
		printIndent();
		oss << "__device_" << var_decl->name << " = ";
		var_decl->initVal->accept(this);
		oss << ";" << std::endl;
	} else {
		printIndent();
		oss << "cudaMalloc(&__device_" << var_decl->name << ", ";
		if (size_expr != nullptr)
			size_expr->accept(this);
		else
			oss << vector_type->range_indexset;
		oss << " * sizeof(";
		vector_type->vector_element_type->accept(this);
		oss << "));" << std::endl;
	}
	
	printIndent();
	oss << "cudaMemcpyToSymbol(";
	oss << var_decl->name;
	oss << ", &__device_" << var_decl->name << ", sizeof(";
	vector_type->vector_element_type->accept(this);	
	oss << "*), 0);" << std::endl;

	printIndent();
	oss << "__host_" << var_decl->name << " = new ";
	vector_type->vector_element_type->accept(this);
	oss << "[";
	if (size_expr != nullptr)
		size_expr->accept(this);
	else
		oss << vector_type->range_indexset;
	oss << "];" << std::endl;
	
		
}
void KernelVariableExtractor::visit(mir::VarExpr::Ptr var_expr) {
	if (mir_context_->isLoweredConst(var_expr->var.getName())) {
		return;
	}
	
	insertVar(var_expr->var);
}
void KernelVariableExtractor::visit(mir::VarDecl::Ptr var_decl) {
	insertDecl(var_decl);
}
void KernelVariableExtractor::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr esae) {
	mir::MIRVisitor::visit(esae);
	hoisted_pqs.push_back(esae->priority_queue_used);	
}
void CodeGenGPU::genFusedWhileLoop(mir::WhileStmt::Ptr while_stmt) {
	
	// First we generate a unique function name for this fused kernel
	std::string fused_kernel_name = "fused_kernel_body_" + mir_context_->getUniqueNameCounterString();
	while_stmt->fused_kernel_name = fused_kernel_name;

	// Now we extract the list of variables that are used in the kernel that are not const 
	// So we can hoist them
	KernelVariableExtractor extractor(mir_context_);
	while_stmt->accept(&extractor);

	while_stmt->hoisted_vars = extractor.hoisted_vars;
	while_stmt->hoisted_decls = extractor.hoisted_decls;
	
	CodeGenGPUFusedKernel codegen (oss, mir_context_, module_name, "");
	codegen.current_while_stmt = while_stmt;
	
	oss << "// ";
	for (auto var: extractor.hoisted_vars) 
		oss << var.getName() << " ";
	oss << std::endl;
	
	for (auto var: extractor.hoisted_vars) {	
		var.getType()->accept(this);	
		oss << " __device__ " << fused_kernel_name << "_" << var.getName() << ";" << std::endl;
	}
	codegen.kernel_hoisted_vars = extractor.hoisted_vars;
	codegen.current_kernel_name = fused_kernel_name;
	for (auto var: extractor.hoisted_pqs)
		codegen.kernel_hoisted_vars.push_back(var);

	oss << "void __global__ " << fused_kernel_name << "(void) {" << std::endl;	
	codegen.indent();
	codegen.printIndent();
	oss << "grid_group _grid = this_grid();" << std::endl;
	codegen.printIndent();
	oss << "int32_t _thread_id = threadIdx.x + blockIdx.x * blockDim.x;" << std::endl;
	// For all the variables we would also generate local copies in each thread
	for (auto var: extractor.hoisted_vars) {	
		codegen.printIndent();
		oss << "auto __local_" << var.getName() << " = " << fused_kernel_name << "_" << var.getName() << ";" << std::endl;
	}
	for (auto var: extractor.hoisted_pqs) {
		codegen.printIndent();
		oss << "auto __local_" << var.getName() << " = " << var.getName() << ";" << std::endl;	
	}
	
	codegen.printIndent();
	oss << "while (";
	while_stmt->cond->accept(&codegen);
	oss << ") {" << std::endl;
	codegen.indent();
	while_stmt->body->accept(&codegen);
	codegen.dedent();
	codegen.printIndent();
	oss << "}" << std::endl;

	// After the kernel has ended, we should copy back all the variables
	codegen.printIndent();
	oss << "if (_thread_id == 0) {" << std::endl;
	codegen.indent();
	for (auto var: extractor.hoisted_vars) {	
		codegen.printIndent();
		oss << fused_kernel_name << "_" << var.getName() << " = " << "__local_" << var.getName() << ";" << std::endl;
	}
	for (auto var: extractor.hoisted_pqs) {
		codegen.printIndent();
		oss << var.getName() << " = __local_" << var.getName() << ";" << std::endl;
	}
	codegen.dedent();
	codegen.printIndent();
	oss << "}" << std::endl;	
	codegen.dedent();
	codegen.printIndent();
	oss << "}" << std::endl;			

	codegen.kernel_hoisted_vars.clear();
}
void CodeGenGPUFusedKernel::visit(mir::StmtBlock::Ptr stmt_block) {
	for (auto stmt : *(stmt_block->stmts)) {
		stmt->accept(this);
	}
}
void CodeGenGPU::genFuncDecl(mir::FuncDecl::Ptr func_decl) {
	if (func_decl->result.isInitialized()) {
		func_decl->result.getType()->accept(this);
	} else {
		oss << "void";
	}

	if (func_decl->function_context & mir::FuncDecl::function_context_type::CONTEXT_DEVICE)
		oss << " " << "__device__" << " " << func_decl->name << "(";
	else
		oss << " " << func_decl->name << "(";

	bool printDelimeter = false;
	for (auto arg: func_decl->args) {
		if (printDelimeter)
			oss << ", ";
		arg.getType()->accept(this);
		oss << " " << arg.getName();
		printDelimeter = true;
	}
	oss << ");" << std::endl;
}
void CodeGenGPUKernelEmitter::visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr) {

	// First we generate the function that is passed to the load balancing function

	std::string load_balancing_arg = "gpu_operator_body_" + mir_context_->getUniqueNameCounterString();

	oss << "template <typename EdgeWeightType>" << std::endl;
	oss << "void __device__ " << load_balancing_arg << "(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {" << std::endl;
	indent();
	printIndent();
	oss << "// Body of the actual operator code" << std::endl;
	if (apply_expr->to_func && apply_expr->to_func->function_name->name != "") {
		printIndent();
		oss << "if (!" << apply_expr->to_func->function_name->name << "(dst))" << std::endl;
		indent();
		printIndent();
		oss << "return;" << std::endl;
		dedent();
	}
	mir::FuncDecl::Ptr input_function = mir_context_->getFunction(apply_expr->input_function->function_name->name);
	// Enqueueing is disabled from here. We are now enqueing from the UDF 
	if (apply_expr->is_weighted) {	
		printIndent();
		oss << "EdgeWeightType weight = graph.d_edge_weight[edge_id];" << std::endl;
		printIndent();
		oss << apply_expr->input_function->function_name->name << "(src, dst, weight";
	} else {
		printIndent();
		oss << apply_expr->input_function->function_name->name << "(src, dst";
	}
	if (apply_expr->requires_output)
		oss << ", output_frontier";
	oss << ");" << std::endl;
	dedent();
	printIndent();
	oss << "}" << std::endl;	
	apply_expr->device_function = load_balancing_arg;
	
}

void CodeGenGPUKernelEmitter::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr apply_expr) {



	if (apply_expr->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PUSH) {
		// First we generate the function that is passed to the load balancing function
		std::string load_balancing_arg = "gpu_operator_body_" + mir_context_->getUniqueNameCounterString();

		oss << "template <typename EdgeWeightType>" << std::endl;
		oss << "void __device__ " << load_balancing_arg << "(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {" << std::endl;
		indent();
		printIndent();
		oss << "// Body of the actual operator code" << std::endl;
		if (apply_expr->to_func && apply_expr->to_func->function_name->name != "") {
			printIndent();
			oss << "if (!" << apply_expr->to_func->function_name->name << "(dst))" << std::endl;
			indent();
			printIndent();
			oss << "return;" << std::endl;
			dedent();
		}
		mir::FuncDecl::Ptr input_function = mir_context_->getFunction(apply_expr->input_function->function_name->name);
		// Enqueueing is disabled from here. We are now enqueing from the UDF 
		if (apply_expr->is_weighted) {	
			printIndent();
			oss << "EdgeWeightType weight = graph.d_edge_weight[edge_id];" << std::endl;
			printIndent();
			oss << apply_expr->input_function->function_name->name << "(src, dst, weight";
		} else {
			printIndent();
			oss << apply_expr->input_function->function_name->name << "(src, dst";
		}
		if (apply_expr->requires_output)
			oss << ", output_frontier";
		oss << ");" << std::endl;
		dedent();
		printIndent();
		oss << "}" << std::endl;	
		apply_expr->device_function = load_balancing_arg;	
	} else if (apply_expr->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL) {
		// First we generate the function that is passed to the load balancing function
		std::string load_balancing_arg = "gpu_operator_body_" + mir_context_->getUniqueNameCounterString();
		
		oss << "template <typename EdgeWeightType>" << std::endl;
		oss << "void __device__ " << load_balancing_arg << "(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {" << std::endl;
		indent();
		printIndent();
		oss << "// Body of the actual operator" << std::endl;
		// Before we generate the call to the UDF, we have to check if the dst is on the input frontier
		if (apply_expr->from_func && apply_expr->from_func->function_name->name != "") {	
			if (apply_expr->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BOOLMAP) {
				printIndent();
				oss << "if (!input_frontier.d_byte_map_input[dst])" << std::endl;
				indent();
				printIndent();
				oss << "return;" << std::endl;
				dedent();
			} else if (apply_expr->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BITMAP) {
				printIndent();
				oss << "if (!gpu_runtime::checkBit(input_frontier.d_bit_map_input, dst))" << std::endl;
				indent();
				printIndent();
				oss << "return;" << std::endl;
				dedent();
			}
		}

		mir::FuncDecl::Ptr input_function = mir_context_->getFunction(apply_expr->input_function->function_name->name);
		// Enqueueing is disabled from here. We are now enqueing from the UDF 
		if (apply_expr->is_weighted) {	
			printIndent();
			oss << "EdgeWeightType weight = graph.d_edge_weight[edge_id];" << std::endl;
			printIndent();
			oss << apply_expr->input_function->function_name->name << "(dst, src, weight";
		} else {
			printIndent();
			oss << apply_expr->input_function->function_name->name << "(dst, src";
		}
		if (apply_expr->requires_output)
			oss << ", output_frontier";
		oss << ");" << std::endl;
		dedent();
		printIndent();
		oss << "}" << std::endl;	
		apply_expr->device_function = load_balancing_arg;
	}
}

void CodeGenGPUKernelEmitter::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {

	// First we generate the function that is passed to the load balancing function
	std::string load_balancing_arg = "gpu_operator_body_" + mir_context_->getUniqueNameCounterString();
	
	oss << "template <typename EdgeWeightType>" << std::endl;
	oss << "void __device__ " << load_balancing_arg << "(gpu_runtime::GraphT<EdgeWeightType> graph, int32_t src, int32_t dst, int32_t edge_id, gpu_runtime::VertexFrontier input_frontier, gpu_runtime::VertexFrontier output_frontier) {" << std::endl;
	indent();
	printIndent();
	oss << "// Body of the actual operator" << std::endl;
	// Before we generate the call to the UDF, we have to check if the dst is on the input frontier
	if (apply_expr->from_func && apply_expr->from_func->function_name->name != "") {	
		if (apply_expr->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BOOLMAP) {
			printIndent();
			oss << "if (!input_frontier.d_byte_map_input[dst])" << std::endl;
			indent();
			printIndent();
			oss << "return;" << std::endl;
			dedent();
		} else if (apply_expr->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BITMAP) {
			printIndent();
			oss << "if (!gpu_runtime::checkBit(input_frontier.d_bit_map_input, dst))" << std::endl;
			indent();
			printIndent();
			oss << "return;" << std::endl;
			dedent();
		}
	}

	mir::FuncDecl::Ptr input_function = mir_context_->getFunction(apply_expr->input_function->function_name->name);
	// Enqueueing is disabled from here. We are now enqueing from the UDF 
	if (apply_expr->is_weighted) {	
		printIndent();
		oss << "EdgeWeightType weight = graph.d_edge_weight[edge_id];" << std::endl;
		printIndent();
		oss << apply_expr->input_function->function_name->name << "(dst, src, weight";
	} else {
		printIndent();
		oss << apply_expr->input_function->function_name->name << "(dst, src";
	}
	if (apply_expr->requires_output)
		oss << ", output_frontier";
	oss << ");" << std::endl;
	dedent();
	printIndent();
	oss << "}" << std::endl;	
	apply_expr->device_function = load_balancing_arg;

}

void CodeGenGPU::genIncludeStmts(void) {
	oss << "#include \"gpu_intrinsics.h\"" << std::endl;
	oss << "#include <cooperative_groups.h>" << std::endl;
	oss << "using namespace cooperative_groups;" << std::endl;
}

void CodeGenGPU::genGlobalDeclarations(void) {
	for (auto stmt: mir_context_->hybrid_gpu_stmts) {
		std::string threshold_var_name = "hybrid_threshold_var" + mir_context_->getUniqueNameCounterString();	
		oss << "float " << threshold_var_name << ";" << std::endl;
		oss << "float __device__ __device_" << threshold_var_name << ";" << std::endl;
		stmt->threshold_var_name = threshold_var_name;
	}
	oss << "int32_t __delta_param;" << std::endl;	
}

void CodeGenGPU::genEdgeSets(void) {
	for (auto edgeset: mir_context_->getEdgeSets()) {
		auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
		edge_set_type->accept(this);
		oss << " __device__ " << edgeset->name << ";" << std::endl;
		edge_set_type->accept(this);
		oss << " " << "__host_" << edgeset->name << ";" << std::endl;

		bool requires_transpose = false;
		bool requires_blocking = false;
		uint32_t blocking_size = 0;
		if (mir_context_->graphs_with_blocking.find(edgeset->name) != mir_context_->graphs_with_blocking.end()) {
			blocking_size = mir_context_->graphs_with_blocking[edgeset->name];
			auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
			edge_set_type->accept(this);
			oss << " __device__ " << edgeset->name << "__blocked_" << blocking_size << ";" << std::endl;
			edge_set_type->accept(this);
			oss << " " << "__host_" << edgeset->name << "__blocked_" << blocking_size << ";" << std::endl;
			requires_blocking = true;
		}

		if (mir_context_->graphs_with_transpose.find(edgeset->name) != mir_context_->graphs_with_transpose.end() && mir_context_->graphs_with_transpose[edgeset->name]) {
			auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
			edge_set_type->accept(this);
			oss << " __device__ " << edgeset->name << "__transposed" << ";" << std::endl;
			edge_set_type->accept(this);
			oss << " __host_" << edgeset->name << "__transposed" << ";" << std::endl;
			requires_transpose = true;
			
		}
		if (requires_transpose && requires_blocking) {
			auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
			edge_set_type->accept(this);
			oss << " __device__ " << edgeset->name << "__blocked_" << blocking_size << "__transposed" << ";" << std::endl;
			edge_set_type->accept(this);
			oss << " __host_" << edgeset->name << "__blocked_" << blocking_size << "__transposed" << ";" << std::endl;
		}
		
		
	}
}

void CodeGenGPU::visit(mir::EdgeSetType::Ptr edgeset_type) {
	if (edgeset_type->weight_type != nullptr) {
		oss << "gpu_runtime::GraphT<";
		edgeset_type->weight_type->accept(this);
		oss << ">";	
	} else {
		oss << "gpu_runtime::GraphT<char>";
	}
}

void CodeGenGPU::visit(mir::PriorityQueueType::Ptr pqt) {
	oss << "gpu_runtime::GPUPriorityQueue<";
	pqt->priority_type->accept(this);
	oss << ">";
}

void CodeGenGPU::visit(mir::VertexSetType::Ptr vertexset_type) {
	oss << "gpu_runtime::VertexFrontier";
}
void CodeGenGPU::visit(mir::ListType::Ptr list_type) {
	if (mir::isa<mir::VertexSetType>(list_type->element_type)) {
		oss << "gpu_runtime::VertexFrontierList";
		return;
	}
	oss << "std::vector<";
	list_type->element_type->accept(this);
	oss << ">";
}
void CodeGenGPU::visit(mir::ListAllocExpr::Ptr alloc_expr) {
	if (mir::isa<mir::VertexSetType>(alloc_expr->element_type)) {
		oss << "gpu_runtime::create_new_vertex_frontier_list(";		
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
void CodeGenGPU::visit(mir::ScalarType::Ptr scalar_type) {
	switch(scalar_type->type) {
		case mir::ScalarType::Type::INT:
			oss << "int32_t";
			break;
		case mir::ScalarType::Type::UINT:
			oss << "uint32_t";
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
		case mir::ScalarType::Type::COMPLEX:
			assert(false && "Complex type not yet supported with the GPU backend\n");
			break;
		case mir::ScalarType::Type::STRING:
			assert(false && "String type not yet supported with the GPU backend\n");
			break;
		default:
			assert(false && "Invalid type enum for scalar type\n");
			break;
	}
}

void CodeGenGPU::genHybridThresholds(void) {
	for (auto stmt: mir_context_->hybrid_gpu_stmts) {
		std::string var_name = stmt->threshold_var_name;
		if (stmt->threshold < 0) {
			printIndent();
			oss << stmt->threshold_var_name << " = gpu_runtime::str_to_float(argv[" << stmt->argv_index << "]);" << std::endl;
		} else {
			printIndent();
			oss << stmt->threshold_var_name << " = " << stmt->threshold << ";" << std::endl;
		}
		printIndent();
		oss << "cudaMemcpyToSymbol(__device_" << stmt->threshold_var_name << ", &" << stmt->threshold_var_name << ", sizeof(float), 0);" << std::endl;
	}
}
void CodeGenGPU::visit(mir::FuncDecl::Ptr func_decl) {
	if (func_decl->type == mir::FuncDecl::Type::EXTERNAL) {
		assert(false && "GPU backend currently doesn't support external functions\n");
	} else {
		// First generate the signature of the function
		if (func_decl->name == "main") {
			oss << "int " << getBackendFunctionLabel() << " main(int argc, char* argv[])";
		} else {
			if (func_decl->result.isInitialized()) {
				func_decl->result.getType()->accept(this);
			} else {
				oss << "void";
			}
			oss << " " << getBackendFunctionLabel() << " " << func_decl->name << "(";
			bool printDelimeter = false;
			for (auto arg: func_decl->args) {
				if (printDelimeter)
					oss << ", ";
				arg.getType()->accept(this);
				oss << " " << arg.getName();
				printDelimeter = true;
			}
			oss << ")";	
		}
		oss << " {" << std::endl;
		indent();

		if (func_decl->name == "main") {
			genHybridThresholds();
			if (mir_context_->delta_ <= 0) {
				printIndent();
				oss << "__delta_param = gpu_runtime::str_to_int(argv[" << - mir_context_->delta_ << "]);" << std::endl;
			} else {
				printIndent();
				oss << "__delta_param = " << mir_context_->delta_ << ";" << std::endl;
			}
			for (auto stmt: mir_context_->edgeset_alloc_stmts) {
				mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(stmt);
				mir::EdgeSetLoadExpr::Ptr edge_set_load_expr = mir::to<mir::EdgeSetLoadExpr>(assign_stmt->expr);
				mir::VarExpr::Ptr lhs_var = mir::to<mir::VarExpr>(assign_stmt->lhs);
				std::string var_name  = lhs_var->var.getName();
				
				printIndent();
				oss << "gpu_runtime::load_graph(";
				oss << "__host_" << var_name << ", ";
				edge_set_load_expr->file_name->accept(this);
				oss << ", false);" << std::endl;

				printIndent();
				oss << "cudaMemcpyToSymbol(";
				oss << var_name << ", &__host_" << var_name << ", sizeof(__host_" << var_name << "), 0, cudaMemcpyHostToDevice);" << std::endl;
				bool requires_blocking = false;
				bool requires_transpose = false;
				uint32_t blocking_size = 0;
				if (mir_context_->graphs_with_blocking.find(var_name) != mir_context_->graphs_with_blocking.end()) {
					blocking_size = mir_context_->graphs_with_blocking[var_name];		
					requires_blocking = true;
					printIndent();
					oss << "gpu_runtime::block_graph_edges(__host_" << var_name << ", __host_" << var_name << "__blocked_" << blocking_size << ", " << blocking_size << ");" << std::endl;
					printIndent();
					oss << "cudaMemcpyToSymbol(";
					oss << var_name << "__blocked_" << blocking_size << ", &__host_" << var_name << "__blocked_" << blocking_size << ", sizeof(__host_" << var_name << "__blocked_" << blocking_size << "), 0, cudaMemcpyHostToDevice);" << std::endl;
				}

				if (mir_context_->graphs_with_transpose.find(var_name) != mir_context_->graphs_with_transpose.end() && mir_context_->graphs_with_transpose[var_name]) {
					requires_transpose = true;
					printIndent();
					oss << "__host_" << var_name << "__transposed = gpu_runtime::builtin_transpose(__host_" << var_name << ");" << std::endl;
					printIndent();
					oss << "cudaMemcpyToSymbol(";
					oss << var_name << "__transposed" << ", &__host_" << var_name << "__transposed, sizeof(__host_" << var_name << "__transposed), 0, cudaMemcpyHostToDevice);" << std::endl;
				}
				if (requires_transpose && requires_blocking) {
					printIndent();
					oss << "gpu_runtime::block_graph_edges(__host_" << var_name << "__transposed, __host_" << var_name << "__blocked_" << blocking_size << "__transposed, " << blocking_size << ");" << std::endl;
					printIndent();
					oss << "cudaMemcpyToSymbol(";
					oss << var_name << "__blocked_" << blocking_size << "__transposed, &__host_" << var_name << "__blocked_" << blocking_size << "__transposed, sizeof(__host_" << var_name << "__blocked_" << blocking_size << "__transposed), 0, cudaMemcpyHostToDevice);" << std::endl;
					
				}
				

			}
			for (auto constant: mir_context_->getLoweredConstants()) {
				if (mir::isa<mir::VectorType>(constant->type)) {
					if (constant->needs_allocation) 
						genPropertyArrayAlloca(constant);
				} else {
					if (constant->initVal != nullptr) {
						printIndent();
						oss << "__host_" << constant->name << " = ";
						constant->initVal->accept(this);
						oss << ";" << std::endl;
						printIndent();
						oss << "cudaMemcpyToSymbol(" << constant->name << ", &__host_" << constant->name << ", sizeof(";
						constant->type->accept(this);
						oss << "), 0, cudaMemcpyHostToDevice);" << std::endl;
					}
				}
				if (mir::isa<mir::PriorityQueueType>(constant->type)) {
					printIndent();
					oss << "cudaGetSymbolAddress(((void**)&__device_" << constant->name << "), " << constant->name << ");" << std::endl;
				}
			}
			for (auto stmt: mir_context_->field_vector_init_stmts) {
				stmt->accept(this);
			}
		}
		if (func_decl->body && func_decl->body->stmts) {
			if (func_decl->result.isInitialized()) {
				printIndent();
				func_decl->result.getType()->accept(this);
				oss << " " << func_decl->result.getName() << ";" << std::endl;
			}	
			func_decl->body->accept(this);	
			if (func_decl->result.isInitialized()) {
				printIndent();
				oss << "return " << func_decl->result.getName() << ";" << std::endl;
			}
		}	
		
		dedent();
		printIndent();
		oss << "}" << std::endl;
	}
}
void CodeGenGPU::visit(mir::ElementType::Ptr element_type) {
	oss << "int32_t";
}
void CodeGenGPU::genPriorityUpdateOperator(mir::PriorityUpdateOperator::Ptr puo) {
	printIndent();
	oss << "if (";
	if (mir::isa<mir::PriorityUpdateOperatorMin>(puo)) {
		mir::PriorityUpdateOperatorMin::Ptr puom = mir::to<mir::PriorityUpdateOperatorMin>(puo);
		if (puom->is_atomic) {
			oss << "gpu_runtime::writeMin";
		} else {
			assert(false && "Currently only atomic priority update is supported");
		}
		oss << "(";
		oss << "&(";
		//puom->priority_queue->accept(this);
		oss << "__output_frontier.d_priority_array[";
		puom->destination_node_id->accept(this);
		oss << "]), ";
		puom->new_val->accept(this);
		oss << ")";
	}
	oss << " && ";
	oss << "__output_frontier.d_priority_array[";
	puo->destination_node_id->accept(this);
	oss << "] < (";
	//puo->priority_queue->accept(this);
	oss << "__output_frontier.priority_cutoff)";
	oss << ") {" << std::endl;
	indent();

	mir::UpdatePriorityEdgeSetApplyExpr::Ptr upesae = puo->edgeset_apply_expr;	
	mir::EnqueueVertex::Ptr evp = std::make_shared<mir::EnqueueVertex>();
	evp->vertex_id = puo->destination_node_id;
	mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(puo->priority_queue);
	// Since this variable is created temporarily, we don;t need type
	mir::Var var("__output_frontier", nullptr);
	mir::VarExpr::Ptr frontier_expr = std::make_shared<mir::VarExpr>();
	frontier_expr->var = var;	
	
	evp->vertex_frontier = frontier_expr;
	if (upesae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
		evp->type = mir::EnqueueVertex::Type::SPARSE;
	} else if (upesae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BOOLMAP) {
		evp->type = mir::EnqueueVertex::Type::BOOLMAP;
	} else if (upesae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BITMAP) {
		evp->type = mir::EnqueueVertex::Type::BITMAP;
	} 
	
	evp->accept(this);
	dedent();
	printIndent();	
	oss << "}" << std::endl;

}
void CodeGenGPU::visit(mir::ExprStmt::Ptr expr_stmt) {
	if (mir::isa<mir::EdgeSetApplyExpr>(expr_stmt->expr)) {
		genEdgeSetApplyExpr(mir::to<mir::EdgeSetApplyExpr>(expr_stmt->expr), nullptr);
	} else if (mir::isa<mir::PriorityUpdateOperatorMin>(expr_stmt->expr)) {
		genPriorityUpdateOperator(mir::to<mir::PriorityUpdateOperatorMin>(expr_stmt->expr));
	} else {
		printIndent();
		expr_stmt->expr->accept(this);
		oss << ";" << std::endl;
	}
}

void CodeGenGPU::visit(mir::VarExpr::Ptr var_expr) {
	if (is_hoisted_var(var_expr->var)) {
		oss << "__local_" << var_expr->var.getName();
		return;
	} else
		oss << var_expr->var.getName();
}
void CodeGenGPUHost::visit(mir::VarExpr::Ptr var_expr) {
	if (mir_context_->isLoweredConst(var_expr->var.getName())) {
		oss << "__host_" << var_expr->var.getName();
		return;
	} else 
		oss << var_expr->var.getName();

}
void CodeGenGPUFusedKernel::visit(mir::VarExpr::Ptr var_expr) {
	if (is_hoisted_var(var_expr->var)) {
		oss << "__local_" << var_expr->var.getName();
		return;
	} else 
		oss << var_expr->var.getName();
}
void CodeGenGPU::genEdgeSetApplyExpr(mir::EdgeSetApplyExpr::Ptr esae, mir::Expr::Ptr target) {
	if (target != nullptr && (esae->from_func == nullptr || esae->from_func->function_name->name == "")) {
		assert(false && "GPU backend doesn't currently support creating output frontier without input frontier\n");
	}		
	// We will assume that the output frontier can reuse the input frontier. 
	// Assert that the frontier can be reused
	/*
	if (target != nullptr && esae->frontier_reusable != true) {
		assert(false && "GPU backend currently doesn't support creating frontiers from the apply expressions. Could not find opportunity for reuse\n");
	}
	*/

	printIndent();
	oss << "{" << std::endl;
	indent();
	
	std::string load_balance_function = "gpu_runtime::vertex_based_load_balance";
	if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::TWCE) {
		load_balance_function = "gpu_runtime::TWCE_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::EDGE_ONLY) {
		load_balance_function = "gpu_runtime::edge_only_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::TWC) {
		load_balance_function = "gpu_runtime::TWC_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::CM) {
		load_balance_function = "gpu_runtime::CM_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::WM) {
		load_balance_function = "gpu_runtime::WM_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::STRICT) {
		load_balance_function = "gpu_runtime::strict_load_balance";
	}

	if (mir::isa<mir::PushEdgeSetApplyExpr>(esae) || mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(esae) && esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PUSH) {
		if (esae->from_func && esae->from_func->function_name->name != "") {
			printIndent();
			oss << "gpu_runtime::vertex_set_prepare_sparse(";
			oss << esae->from_func->function_name->name;
			oss << ");" << std::endl;
		}
	} else if (mir::isa<mir::PullEdgeSetApplyExpr>(esae) || mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(esae) && esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL) {
		if (esae->from_func && esae->from_func->function_name->name != "") {
			if (esae->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BOOLMAP) {
				printIndent();
				oss << "gpu_runtime::vertex_set_prepare_boolmap(";
				oss << esae->from_func->function_name->name;
				oss << ");" << std::endl;
			} else if (esae->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BITMAP) {
				printIndent();
				oss << "gpu_runtime::vertex_set_prepare_bitmap(";
				oss << esae->from_func->function_name->name;
				oss << ");" << std::endl;
			}
		}

		std::string to_func ;
		if (esae->to_func)
			to_func = esae->to_func->function_name->name;
		else 
			to_func = "";
		if (to_func != "") {
			printIndent();
			oss << "gpu_runtime::vertex_set_create_reverse_sparse_queue_host<" << to_func << ">(";
			oss << esae->from_func->function_name->name << ");" << std::endl;
		}

	}

	// We will have to create a new frontier in case the frontier cannot be reused
	// If the frontier is reusable, we simply assign the old to the new
	if (target != nullptr) {
		if (esae->frontier_reusable) {
			printIndent();
			target->accept(this);
			oss << " = " << esae->from_func->function_name->name << ";" << std::endl;
		} else {
			printIndent();
			target->accept(this);
			oss << " = ";
			oss << "gpu_runtime::create_new_vertex_set(gpu_runtime::builtin_getVertices(";
			esae->target->accept(this);
			oss << "), 0);" << std::endl;
		}
	}
	if (mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(esae)) {
		mir::UpdatePriorityEdgeSetApplyExpr::Ptr upesae = mir::to<mir::UpdatePriorityEdgeSetApplyExpr>(esae);
		printIndent();
		oss << "cudaMemcpyToSymbol(" << upesae->priority_queue_used.getName() << ", &__host_" << upesae->priority_queue_used.getName() << ", sizeof(" << upesae->priority_queue_used.getName() << "), 0);" << std::endl;
	}

	// Before the load balance if the update requires dedup, then update the counters
	if (esae->fused_dedup && target != nullptr) {
		printIndent();
		target->accept(this);
		oss << ".curr_dedup_counter++;" << std::endl;
	}	
	printIndent();
	oss << load_balance_function << "_host<";

	mir::Var target_var = mir::to<mir::VarExpr>(esae->target)->var;
	mir::EdgeSetType::Ptr target_type = mir::to<mir::EdgeSetType>(target_var.getType());
	if (target_type->weight_type == nullptr)
		oss << "char";
	else
		target_type->weight_type->accept(this);

	std::string accessor_type = "gpu_runtime::AccessorSparse";
	if (!esae->from_func || esae->from_func->function_name->name == "")
		accessor_type = "gpu_runtime::AccessorAll";
	if (esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL && (esae->from_func == nullptr || esae->to_func->function_name->name == ""))
		accessor_type = "gpu_runtime::AccessorAll";
	std::string src_filter = "gpu_runtime::true_function";
	if (esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL && esae->from_func && esae->to_func->function_name->name != "")
		src_filter = esae->to_func->function_name->name;

	oss << ", " << esae->device_function << ", " << accessor_type << ", " << src_filter << ">(";
	esae->target->accept(this);
	if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::EDGE_ONLY && esae->applied_schedule.edge_blocking == fir::gpu_schedule::SimpleGPUSchedule::edge_blocking_type::BLOCKED) {
		oss << "__blocked_" << esae->applied_schedule.edge_blocking_size;
	}
	if (esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL) {
		oss << "__transposed";
	}
	oss << ", ";
	if (esae->from_func && esae->from_func->function_name->name != "")
		oss << esae->from_func->function_name->name;
	else {
		esae->target->accept(this);
		oss << ".getFullFrontier()";
	}
	oss << ", ";
	if (target != nullptr)
		target->accept(this);
	else 
		oss << "gpu_runtime::sentinel_frontier";
	oss << ");" << std::endl;


	printIndent();
	oss << "cudaDeviceSynchronize();" << std::endl;
	if (target != nullptr) {
		if (esae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
			printIndent();
			oss << "gpu_runtime::swap_queues(";
			target->accept(this);
			oss << ");" << std::endl;
			printIndent();
			target->accept(this);
			oss << ".format_ready = gpu_runtime::VertexFrontier::SPARSE;" << std::endl;
		} else if (esae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BITMAP) {
			printIndent();
			oss << "gpu_runtime::swap_bitmaps(";
			target->accept(this);
			oss << ");" << std::endl;
			printIndent();
			target->accept(this);
			oss << ".format_ready = gpu_runtime::VertexFrontier::BITMAP;" << std::endl;
		} else if (esae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BOOLMAP) {
			printIndent();
			oss << "gpu_runtime::swap_bytemaps(";
			target->accept(this);
			oss << ");" << std::endl;
			printIndent();
			target->accept(this);
			oss << ".format_ready = gpu_runtime::VertexFrontier::BYTEMAP;" << std::endl;
		}
	}
	dedent();
	printIndent();
	oss << "}" << std::endl;

}
void CodeGenGPUFusedKernel::genEdgeSetApplyExpr(mir::EdgeSetApplyExpr::Ptr esae, mir::Expr::Ptr target) {
	if (target != nullptr && (esae->from_func == nullptr || esae->from_func->function_name->name == "")) {
		assert(false && "GPU backend doesn't currently support creating output frontier without input frontier\n");
	}
	printIndent();
	oss << "{" << std::endl;
	indent();
	std::string load_balance_function = "gpu_runtime::vertex_based_load_balance";
	if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::TWCE) {
		load_balance_function = "gpu_runtime::TWCE_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::EDGE_ONLY) {
		load_balance_function = "gpu_runtime::edge_only_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::TWC) {
		load_balance_function = "gpu_runtime::TWC_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::CM) {
		load_balance_function = "gpu_runtime::CM_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::WM) {
		load_balance_function = "gpu_runtime::WM_load_balance";
	} else if (esae->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::STRICT) {
		load_balance_function = "gpu_runtime::strict_load_balance";
	}
	
	if (mir::isa<mir::PushEdgeSetApplyExpr>(esae) || mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(esae) && esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PUSH) {
		printIndent();
		oss << "gpu_runtime::vertex_set_prepare_sparse_device(";
		oss << var_name(esae->from_func->function_name->name);
		oss << ");" << std::endl;
	} else if (mir::isa<mir::PullEdgeSetApplyExpr>(esae) || mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(esae) && esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL) {
		if (esae->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BOOLMAP) {
			printIndent();
			oss << "gpu_runtime::vertex_set_prepare_boolmap_device(";
			oss << var_name(esae->from_func->function_name->name);
			oss << ");" << std::endl;
		} else if (esae->applied_schedule.pull_frontier_rep == fir::gpu_schedule::SimpleGPUSchedule::pull_frontier_rep_type::BITMAP) {
			printIndent();
			oss << "gpu_runtime::vertex_set_prepare_bitmap_device(";
			oss << var_name(esae->from_func->function_name->name);
			oss << ");" << std::endl;
		}
		std::string to_func;
		if (esae->to_func)
			to_func = esae->to_func->function_name->name;
		else
			to_func = "";
                
		if (to_func != "") {
			printIndent();
			oss << "gpu_runtime::vertex_set_create_reverse_sparse_queue_device<" << to_func << ">(";
			oss << var_name(esae->from_func->function_name->name) << ");" << std::endl;
		}
	}
	if (target != nullptr) {
		printIndent();
		target->accept(this);	
		oss << " = " << var_name(esae->from_func->function_name->name) << ";" << std::endl;
	}
	if (mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(esae)) {
		mir::UpdatePriorityEdgeSetApplyExpr::Ptr upesae = mir::to<mir::UpdatePriorityEdgeSetApplyExpr>(esae);
		insertUsedPq(upesae->priority_queue_used);
	}
	if (mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(esae)) {
/*
		mir::UpdatePriorityEdgeSetApplyExpr::Ptr upesae = mir::to<mir::UpdatePriorityEdgeSetApplyExpr>(esae);
		printIndent();
		oss << "if (_thread_id == 0) {" << std::endl;
		indent();
		printIndent();
		oss << upesae->priority_queue_used.getName() << " = __local_" << upesae->priority_queue_used.getName() << ";" << std::endl;
		dedent();
		printIndent();
		oss << "}" << std::endl;
		printIndent();
		oss << "_grid.sync();" << std::endl;
		//oss << "cudaMemcpyToSymbol(" << upesae->priority_queue_used.getName() << ", &__host_" << upesae->priority_queue_used.getName() << ", sizeof(" << upesae->priority_queue_used.getName() << "), 0);" << std::endl;
*/
	}
	// Before the load balance if the update requires dedup, then update the counters
	if (esae->fused_dedup && target != nullptr) {
		printIndent();
		target->accept(this);
		oss << ".curr_dedup_counter++;" << std::endl;
	}	
	printIndent();
	oss << load_balance_function << "_device<";
	
	mir::Var target_var = mir::to<mir::VarExpr>(esae->target)->var;
	mir::EdgeSetType::Ptr target_type = mir::to<mir::EdgeSetType>(target_var.getType());
	if (target_type->weight_type == nullptr)
		oss << "char";
	else
		target_type->weight_type->accept(this);
	
	std::string accessor_type = "gpu_runtime::AccessorSparse";
	if (esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL && (esae->to_func == nullptr || esae->to_func->function_name->name == ""))
		accessor_type = "gpu_runtime::AccessorAll";
	std::string src_filter = "gpu_runtime::true_function";
	if (esae->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL && esae->to_func && esae->to_func->function_name->name != "")
		src_filter = esae->to_func->function_name->name;

	oss << ", " << esae->device_function << ", " << accessor_type << ", " << src_filter << ">(";
	esae->target->accept(this);
	oss << ", " << var_name(esae->from_func->function_name->name) << ", ";
	if (target != nullptr) 
		target->accept(this);
	else 
		oss << "gpu_runtime::device_sentinel_frontier";
	oss << ");" << std::endl;
	
	if (target != nullptr) {
		mir::VarExpr::Ptr target_expr = mir::to<mir::VarExpr>(target);
		if (esae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
			printIndent();
			oss << "gpu_runtime::swap_queues_device(";
			target->accept(this);
			oss << ");" << std::endl;	
			printIndent();
			target->accept(this);
			oss << ".format_ready = gpu_runtime::VertexFrontier::SPARSE;" << std::endl;
		} else if (esae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BITMAP) {
			printIndent();
			oss << "gpu_runtime::swap_bitmaps_device(";
			target->accept(this);
			oss << ");" << std::endl;
			printIndent();
			target->accept(this);
			oss << ".format_ready = gpu_runtime::VertexFrontier::BITMAP;" << std::endl;
		} else if (esae->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::UNFUSED_BOOLMAP) {
			printIndent();
			oss << "gpu_runtime::swap_bytemaps_device(";
			target->accept(this);
			oss << ");" << std::endl;
			printIndent();
			target->accept(this);
			oss << ".format_ready = gpu_runtime::VertexFrontier::BYTEMAP;" << std::endl;
		}
	}
	dedent();
	printIndent();
	oss << "}" << std::endl;
	
}
void CodeGenGPU::visit(mir::AssignStmt::Ptr assign_stmt) {
	if (mir::isa<mir::EdgeSetApplyExpr>(assign_stmt->expr)) {
		mir::EdgeSetApplyExpr::Ptr esae = mir::to<mir::EdgeSetApplyExpr>(assign_stmt->expr);	
		genEdgeSetApplyExpr(esae, assign_stmt->lhs);
	} else if (mir::isa<mir::PriorityQueueAllocExpr>(assign_stmt->expr)) {
		mir::PriorityQueueAllocExpr::Ptr pqae = mir::to<mir::PriorityQueueAllocExpr>(assign_stmt->expr);	
		printIndent();
		assign_stmt->lhs->accept(this);
		oss << ".init(";
		std::string graph_name = mir_context_->getEdgeSets()[0]->name;	
		oss << "__host_" << graph_name << ", ";
		std::string vector_name = pqae->vector_function;
		if (mir_context_->isLoweredConst(vector_name))
			oss << "__host_" << vector_name;
		else
			oss << vector_name;
		oss << ", ";
		if (mir_context_->isLoweredConst(vector_name))
			oss << "__device_" << vector_name;
		else
			oss << vector_name;
		oss << ", 0, ";
		oss << "__delta_param";
		oss << ", ";
		pqae->starting_node->accept(this);
		oss << ");" << std::endl;	
	} else if(mir::isa<mir::VertexSetWhereExpr>(assign_stmt->expr)) {
                mir::VertexSetWhereExpr::Ptr vswe = mir::to<mir::VertexSetWhereExpr>(assign_stmt->expr);
		if(!mir_context_->isConstVertexSet(vswe->target)) {
			assert(false && "GPU backend currently doesn't support vertex where on non-const sets");
		}
		auto associated_element_type = mir_context_->getElementTypeFromVectorOrSetName(vswe->target);
		assert(associated_element_type != nullptr);
		auto associated_edge_set = mir_context_->getEdgeSetFromElementType(associated_element_type);
		assert(associated_edge_set != nullptr);
		
		printIndent();
		assign_stmt->lhs->accept(this);
		oss << " = ";
		oss << "gpu_runtime::create_new_vertex_set(";
		oss << "__host_" << associated_edge_set->name << ".num_vertices, 0);" << std::endl;
		printIndent();
		oss << "gpu_runtime::vertex_set_where<";
		oss << vswe->input_func << ">";
		oss << "(__host_" << associated_edge_set->name << ".num_vertices, ";
		assign_stmt->lhs->accept(this);
		oss << ");" << std::endl;
				
	} else {
		printIndent();
		assign_stmt->lhs->accept(this);
		oss << " = ";
		assign_stmt->expr->accept(this);
		oss << ";" << std::endl;
	}
}


void CodeGenGPUFusedKernel::visit(mir::AssignStmt::Ptr assign_stmt) {
	if (mir::isa<mir::EdgeSetApplyExpr>(assign_stmt->expr)) {
		mir::EdgeSetApplyExpr::Ptr esae = mir::to<mir::EdgeSetApplyExpr>(assign_stmt->expr);
		genEdgeSetApplyExpr(esae, assign_stmt->lhs);
	} else {
		if (mir::isa<mir::VarExpr>(assign_stmt->lhs) && is_hoisted_var(mir::to<mir::VarExpr>(assign_stmt->lhs)->var)) {
			printIndent();
			assign_stmt->lhs->accept(this);
			oss << " = ";
			assign_stmt->expr->accept(this);
			oss << ";" << std::endl;
		} else {
			printIndent();
			oss << "if (_thread_id == 0) " << std::endl;
			indent();
			printIndent();
			assign_stmt->lhs->accept(this);
			oss << " = ";
			assign_stmt->expr->accept(this);
			oss << ";" << std::endl;	
			dedent();
			printIndent();
			oss << "_grid.sync();" << std::endl;
		}
	}	
}

void CodeGenGPU::generateBinaryExpr(mir::BinaryExpr::Ptr expr, std::string token) {
	oss << "(";
	expr->lhs->accept(this);
	oss << " " << token << " ";
	expr->rhs->accept(this);
	oss << ")";
}
void CodeGenGPU::visit(mir::AddExpr::Ptr expr) {
	generateBinaryExpr(expr, "+");
}
void CodeGenGPU::visit(mir::MulExpr::Ptr expr) {
	generateBinaryExpr(expr, "*");
}
void CodeGenGPU::visit(mir::DivExpr::Ptr expr) {
	generateBinaryExpr(expr, "/");
}
void CodeGenGPU::visit(mir::SubExpr::Ptr expr) {
	generateBinaryExpr(expr, "-");
}
void CodeGenGPU::visit(mir::NegExpr::Ptr expr) {
	if (expr->negate)
		oss << "-";
	oss << "(";
	expr->operand->accept(this);
	oss << ")";
}


void CodeGenGPU::visit(mir::TensorArrayReadExpr::Ptr expr) {
	expr->target->accept(this);
	oss << "[";
	expr->index->accept(this);
	oss << "]";	
}
void CodeGenGPUHost::visit(mir::TensorArrayReadExpr::Ptr expr) {
	mir::VarExpr::Ptr var_expr = mir::to<mir::VarExpr>(expr->target);
	expr->target->accept(this);
	oss << "[";
	expr->index->accept(this);
	oss << "]";
}

void CodeGenGPU::visit(mir::IntLiteral::Ptr expr) {
	oss << expr->val;
}
void CodeGenGPU::visit(mir::FloatLiteral::Ptr expr) {
	oss << "((float)" << expr->val << ")";
}
void CodeGenGPU::visit(mir::StringLiteral::Ptr expr) {
	oss << "\"";
	for (auto ch : expr->val)
		if (iscntrl(ch) || ch == '\\' || ch == '\"' || ch == '\'')
			oss << "\\0" << std::oct << (int)(ch);	
		else
			oss << ch;
	oss << "\"";
}
void CodeGenGPU::visit(mir::ReduceStmt::Ptr reduce_stmt) {
	switch (reduce_stmt->reduce_op_) {
		case mir::ReduceStmt::ReductionOp::SUM:
			printIndent();
			reduce_stmt->lhs->accept(this);
			oss << " += ";
			reduce_stmt->expr->accept(this);
			oss << ";" << std::endl;
			if (reduce_stmt->tracking_var_name_ != "") {
				printIndent();
				oss << reduce_stmt->tracking_var_name_ << " = true;" << std::endl;
			}
			break;
		case mir::ReduceStmt::ReductionOp::MIN:
			printIndent();
			oss << "if ((";
			reduce_stmt->lhs->accept(this);
			oss << ") > (";
			reduce_stmt->expr->accept(this);
			oss << ")) {" << std::endl;
			indent();
			printIndent();
			reduce_stmt->lhs->accept(this);
			oss << " = ";
			reduce_stmt->expr->accept(this);
			oss << ";" << std::endl;

			if (reduce_stmt->tracking_var_name_ != "") {
				printIndent();
				oss << reduce_stmt->tracking_var_name_ << " = true;" << std::endl;
			}
			dedent();
			printIndent();
			oss << "}" << std::endl;
			break;
		case mir::ReduceStmt::ReductionOp::MAX:
			printIndent();
			oss << "if ((";
			reduce_stmt->lhs->accept(this);
			oss << ") < (";
			reduce_stmt->expr->accept(this);
			oss << ")) {" << std::endl;
			indent();
			printIndent();
			reduce_stmt->lhs->accept(this);
			oss << " = ";
			reduce_stmt->expr->accept(this);
			oss << ";" << std::endl;

			if (reduce_stmt->tracking_var_name_ != "") {
				printIndent();
				oss << reduce_stmt->tracking_var_name_ << " = true;" << std::endl;
			}
			dedent();
			printIndent();
			oss << "}" << std::endl;
			break;
		case mir::ReduceStmt::ReductionOp::ATOMIC_MIN:
			printIndent();
			if (reduce_stmt->tracking_var_name_ != "") 
				oss << reduce_stmt->tracking_var_name_ << " = ";
			oss << "gpu_runtime::writeMin(&";
			reduce_stmt->lhs->accept(this);
			oss << ", ";
			reduce_stmt->expr->accept(this);
			oss << ");" << std::endl;
			break;
		case mir::ReduceStmt::ReductionOp::ATOMIC_SUM:
			if (reduce_stmt->tracking_var_name_ != "") {
				printIndent();
				oss << reduce_stmt->tracking_var_name_ << " = true;" << std::endl;
			}
			printIndent();
			oss << "gpu_runtime::writeAdd(&";
			reduce_stmt->lhs->accept(this);
			oss << ", ";
			reduce_stmt->expr->accept(this);
			oss << ");" << std::endl;
			break;
	}	

}

void CodeGenGPU::visit(mir::EnqueueVertex::Ptr enqueue_vertex) {
	printIndent();
	if (enqueue_vertex->type == mir::EnqueueVertex::Type::SPARSE) {
		oss << "gpu_runtime::enqueueVertexSparseQueue";
		if (enqueue_vertex->fused_dedup) {
			oss << "Dedup";
			if (enqueue_vertex->fused_dedup_perfect) {
				oss <<"Perfect";
			}
		}
		oss << "(";
		enqueue_vertex->vertex_frontier->accept(this);
		oss << ".d_sparse_queue_output";
	} else if (enqueue_vertex->type == mir::EnqueueVertex::Type::BOOLMAP) {
		oss << "gpu_runtime::enqueueVertexBytemap(";
		enqueue_vertex->vertex_frontier->accept(this);
		oss << ".d_byte_map_output";
	} else if (enqueue_vertex->type == mir::EnqueueVertex::Type::BITMAP) {
		oss << "gpu_runtime::enqueueVertexBitmap(";
		enqueue_vertex->vertex_frontier->accept(this);
		oss << ".d_bit_map_output";
	}
	oss << ", ";
	enqueue_vertex->vertex_frontier->accept(this);
	oss << ".d_num_elems_output, ";
	enqueue_vertex->vertex_id->accept(this);
	if (enqueue_vertex->type == mir::EnqueueVertex::Type::SPARSE && enqueue_vertex->fused_dedup == true) {
		oss << ", ";
		enqueue_vertex->vertex_frontier->accept(this);	
	}
	oss << ");" << std::endl;	
	
}

void CodeGenGPU::visit(mir::CompareAndSwapStmt::Ptr cas_stmt) {
	printIndent();
	if (cas_stmt->tracking_var_ != "") 
		oss << cas_stmt->tracking_var_ << " = ";
	oss << "gpu_runtime::CAS(&";
	cas_stmt->lhs->accept(this);
	oss << ", ";
	cas_stmt->compare_val_expr->accept(this);
	oss << ", ";
	cas_stmt->expr->accept(this);
	oss << ");" << std::endl;
}
void CodeGenGPU::visit(mir::VarDecl::Ptr var_decl) {
	
	printIndent();
	var_decl->type->accept(this);
	
	oss << " " << var_decl->name;
	
	if (var_decl->initVal != nullptr) {
		// Special case if RHS is a EdgeSetApplyExpr
		oss << " = ";
		var_decl->initVal->accept(this);
		oss << ";" << std::endl;
		
	} else 
		oss << ";" << std::endl;

	if (mir::isa<mir::EdgeSetType>(var_decl->type)) {
		if (mir_context_->graphs_with_transpose.find(var_decl->name) != mir_context_->graphs_with_transpose.end() && mir_context_->graphs_with_transpose[var_decl->name]) {
			printIndent();
			var_decl->type->accept(this);
			oss << " " << var_decl->name << "__transposed = ";
			oss << "gpu_runtime::builtin_transpose(" << var_decl->name << ");" << std::endl;
		}
	}		
	
}
void CodeGenGPUFusedKernel::visit(mir::VarDecl::Ptr var_decl) {
	// Do nothing for variable declarations on kernel only lower the initialization as assignment
	if (var_decl->initVal != nullptr) {
		printIndent();
		oss << "__local_" << var_decl->name << " = ";
		var_decl->initVal->accept(this);
		oss << ";" << std::endl;
	}
}
void CodeGenGPU::visit(mir::VertexSetDedupExpr::Ptr vsde) {
	if (vsde->perfect_dedup)
		oss << "gpu_runtime::dedup_frontier_perfect(";
	else
		oss << "gpu_runtime::dedup_frontier(";
	vsde->target->accept(this);
	oss << ")";
}
void CodeGenGPUFusedKernel::visit(mir::VertexSetDedupExpr::Ptr vsde) {
	oss << "gpu_runtime::dedup_frontier_device(";
	vsde->target->accept(this);
	oss << ")";
}
void CodeGenGPU::visit(mir::BoolLiteral::Ptr bool_literal) {
	oss << bool_literal->val?"true":"false";
}
void CodeGenGPU::visit(mir::ForStmt::Ptr for_stmt) {
	printIndent();
	oss << "for (int32_t " << for_stmt->loopVar << " = ";
	for_stmt->domain->lower->accept(this);
	oss << "; " << for_stmt->loopVar << " < ";
	for_stmt->domain->upper->accept(this);
	oss << "; " << for_stmt->loopVar << "++) {" << std::endl;
	indent();
	for_stmt->body->accept(this);
	dedent();
	printIndent();
	oss << "}" << std::endl;
}
void CodeGenGPU::visit(mir::WhileStmt::Ptr while_stmt) {
	if (while_stmt->is_fused == true) {
		for (auto var: while_stmt->hoisted_vars) {
			bool to_copy = true;
			for (auto decl: while_stmt->hoisted_decls) {
				if (decl->name == var.getName()) {
					to_copy = false;
					break;
				}
			}
			if (!to_copy)
				continue;
			printIndent();
			oss << "cudaMemcpyToSymbol(" << while_stmt->fused_kernel_name << "_" << var.getName() << ", &" << var.getName() << ", sizeof(" << var.getName() << "), 0, cudaMemcpyHostToDevice);" << std::endl;
		}
		for (auto var: while_stmt->used_priority_queues) {
			printIndent();
			oss << "cudaMemcpyToSymbol(" << var.getName() << ", &__host_" << var.getName() << ", sizeof(__host_" << var.getName() << "), 0);" << std::endl;
		}
		printIndent();
		oss << "cudaLaunchCooperativeKernel((void*)" << while_stmt->fused_kernel_name << ", NUM_CTA, CTA_SIZE, gpu_runtime::no_args);" << std::endl;
		for (auto var: while_stmt->used_priority_queues) {
			printIndent();
			oss << "cudaMemcpyFromSymbol(&__host_" << var.getName() << ", " << var.getName() << ", sizeof(__host_" << var.getName() << "), 0);" << std::endl;
		}
		for (auto var: while_stmt->hoisted_vars) {
			bool to_copy = true;
			for (auto decl: while_stmt->hoisted_decls) {
				if (decl->name == var.getName()) {
					to_copy = false;
					break;
				}
			}
			if (!to_copy)
				continue;
			printIndent();
			oss << "cudaMemcpyFromSymbol(&" << var.getName() << ", " << while_stmt->fused_kernel_name << "_" << var.getName() << ", sizeof(" << var.getName() << "), 0, cudaMemcpyDeviceToHost);" << std::endl;
		}
		return;
	}

	ExtractReadWriteSet extractor(mir_context_);
	while_stmt->cond->accept(&extractor);
	
	printIndent();
	oss << "while (";
	while_stmt->cond->accept(this);
	oss << ") {" << std::endl;
	indent();
	for (auto tare: extractor.write_set) {
		generateHostToDeviceCopy(tare);
	}
	while_stmt->body->accept(this);
	for (auto tare: extractor.read_set) {
		generateDeviceToHostCopy(tare);
	}
	dedent();
	printIndent();
	oss << "}" << std::endl;
}
void CodeGenGPU::visit(mir::IfStmt::Ptr if_stmt) {
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
void CodeGenGPUHost::visit(mir::PrintStmt::Ptr print_stmt) {
	printIndent();
	oss << "std::cout << ";
	print_stmt->expr->accept(this);
	oss << " << std::endl;" << std::endl;
}
void CodeGenGPU::visit(mir::PrintStmt::Ptr print_stmt) {
	assert(false && "Cannot print from device function\n");
}
void CodeGenGPUFusedKernel::visit(mir::PrintStmt::Ptr print_stmt) {
	printIndent();
	oss << "if (_thread_id == 0)" << std::endl;
	indent();
	printIndent();
	oss << "gpu_runtime::print(";
	print_stmt->expr->accept(this);
	oss << ");" << std::endl;
	dedent();
	printIndent();
	oss << "_grid.sync();" << std::endl;
}
void CodeGenGPUHost::visit(mir::Call::Ptr call_expr) {
	if (call_expr->name == "dequeue_ready_set" || call_expr->name == "finished") {
		if (call_expr->name == "dequeue_ready_set")
			call_expr->name = "dequeueReadySet";
		mir::VarExpr::Ptr pq_expr = mir::to<mir::VarExpr>(call_expr->args[0]);
		std::string pq_name = pq_expr->var.getName();
		
		oss << "__host_" << pq_name << "." << call_expr->name << "(__device_" << pq_name << ")";
		return;
	}
	if (call_expr->name == "deleteObject" || call_expr->name.substr(0, strlen("builtin_")) == "builtin_")	
		oss << "gpu_runtime::" << call_expr->name << "(";
	else
		oss << call_expr->name << "(";
	
	bool printDelimeter = false;
	for (auto arg: call_expr->args) {
		if (printDelimeter) 
			oss << ", ";
		arg->accept(this);
		printDelimeter = true;
	}	
	oss << ")";
}

void CodeGenGPU::visit(mir::Call::Ptr call_expr) {
	if (call_expr->name == "dequeue_ready_set" || call_expr->name == "finished") {
		if (call_expr->name == "dequeue_ready_set")
			call_expr->name = "dequeueReadySet";
		mir::VarExpr::Ptr pq_expr = mir::to<mir::VarExpr>(call_expr->args[0]);
		pq_expr->accept(this);
		oss << ".device_" << call_expr->name << "()";
		return;
	}
	if (call_expr->name == "deleteObject" || call_expr->name.substr(0, strlen("builtin_")) == "builtin_")	
		oss << "gpu_runtime::device_" << call_expr->name << "(";
	else
		oss << call_expr->name << "(";
	
	bool printDelimeter = false;
	for (auto arg: call_expr->args) {
		if (printDelimeter) 
			oss << ", ";
		arg->accept(this);
		printDelimeter = true;
	}	
	oss << ")";
}

void CodeGenGPU::visit(mir::EqExpr::Ptr eq_expr) {
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
void CodeGenGPU::visit(mir::BreakStmt::Ptr break_stmt) {
	printIndent();
	oss << "break;" << std::endl;
}
void CodeGenGPU::visit(mir::VertexSetApplyExpr::Ptr vsae) {
	auto mir_var = mir::to<mir::VarExpr> (vsae->target);
	if (!mir_context_->isConstVertexSet(mir_var->var.getName())) {
		// This assumes that the parent of the expression is a ExprStmt
		oss << "gpu_runtime::vertex_set_prepare_sparse(";
		oss << mir_var->var.getName(); 
		oss << ");" << std::endl;
		printIndent();
		oss << mir_var->var.getName() << ".format_ready = gpu_runtime::VertexFrontier::SPARSE;" << std::endl;
		printIndent();
	}
	oss << "gpu_runtime::vertex_set_apply_kernel<"; 
	if (mir_context_->isConstVertexSet(mir_var->var.getName())) {
		oss << "gpu_runtime::AccessorAll";
	} else {
		oss << "gpu_runtime::AccessorSparse";
	}
	oss << ", ";
	oss << vsae->input_function->function_name->name << ">";
	oss << "<<<NUM_CTA, CTA_SIZE>>>";
	if (mir_context_->isConstVertexSet(mir_var->var.getName())) {
		auto associated_element_type = mir_context_->getElementTypeFromVectorOrSetName(mir_var->var.getName());
		assert(associated_element_type != nullptr);
		//auto associated_element_type_size = mir_context_->getElementCount(associated_element_type);
		//assert(associated_element_type_size != nullptr);
		auto associated_edge_set = mir_context_->getEdgeSetFromElementType(associated_element_type);
		assert(associated_edge_set != nullptr);
		oss << "(";
		//associated_element_type_size->accept(this);
		oss << "__host_" << associated_edge_set->name << ".getFullFrontier()";
		oss << ")";	
	} else {
		oss << "(";
		oss << mir_var->var.getName();
		oss << ")";
	}		
}
void CodeGenGPUFusedKernel::visit(mir::VertexSetApplyExpr::Ptr vsae) {
	auto mir_var = mir::to<mir::VarExpr> (vsae->target);
	if (!mir_context_->isConstVertexSet(mir_var->var.getName())) {
		// This assumes that the parent of the expression is a ExprStmt
		oss << "gpu_runtime::vertex_set_prepare_sparse_device(";
		oss << var_name(mir_var->var.getName());
		oss << ");" << std::endl;
		printIndent();
		oss << var_name(mir_var->var.getName()) << ".format_ready = gpu_runtime::VertexFrontier::SPARSE;" << std::endl;
		printIndent();
	}
	oss << "gpu_runtime::vertex_set_apply<"; 
	if (mir_context_->isConstVertexSet(mir_var->var.getName())) {
		oss << "gpu_runtime::AccessorAll";
	} else {
		oss << "gpu_runtime::AccessorSparse";
	}
	oss << ", ";
	oss << vsae->input_function->function_name->name << ">";
	if (mir_context_->isConstVertexSet(mir_var->var.getName())) {
		auto associated_element_type = mir_context_->getElementTypeFromVectorOrSetName(mir_var->var.getName());
		assert(associated_element_type != nullptr);
		auto associated_edge_set = mir_context_->getEdgeSetFromElementType(associated_element_type);
		assert(associated_edge_set != nullptr);
		oss << "(";
		oss << var_name(associated_edge_set->name) << ".getFullFrontier()";
		oss << ")";	
	} else {
		oss << "(";
		oss << var_name(mir_var->var.getName());
		oss << ")";
	}		
	oss << ";" << std::endl;
	printIndent();
	oss << "_grid.sync()";
	
}
void CodeGenGPU::visit(mir::VertexSetAllocExpr::Ptr vsae) {
	mir::Expr::Ptr size_expr = mir_context_->getElementCount(vsae->element_type);
	oss << "gpu_runtime::create_new_vertex_set(";
	size_expr->accept(this);
	oss << ", ";
	if (vsae->size_expr == nullptr)
		oss << "0";
	else
		vsae->size_expr->accept(this);
	oss << ")";
}
void CodeGenGPU::generateDeviceToHostCopy(mir::TensorArrayReadExpr::Ptr tare) {
	printIndent();
	mir::Var target = mir::to<mir::VarExpr>(tare->target)->var;
	std::string var_name = target.getName();
	oss << "cudaMemcpy(__host_" << var_name << " + ";
	tare->index->accept(this);
	oss << ", __device_" << var_name << " + ";
	tare->index->accept(this);
	oss << ", sizeof(";
	mir::to<mir::VectorType>(target.getType())->vector_element_type->accept(this);
	oss << "), cudaMemcpyDeviceToHost);" << std::endl;	
	
}
void CodeGenGPU::generateHostToDeviceCopy(mir::TensorArrayReadExpr::Ptr tare) {
	printIndent();
	mir::Var target = mir::to<mir::VarExpr>(tare->target)->var;
	std::string var_name = target.getName();
	oss << "cudaMemcpy(__device_" << var_name << " + ";
	tare->index->accept(this);
	oss << ", __host_" << var_name << " + ";
	tare->index->accept(this);
	oss << ", sizeof(";
	mir::to<mir::VectorType>(target.getType())->vector_element_type->accept(this);
	oss << "), cudaMemcpyHostToDevice);" << std::endl;	
}
void CodeGenGPUHost::visit(mir::StmtBlock::Ptr stmt_block) {
	for (auto stmt: *(stmt_block->stmts)) {
		ExtractReadWriteSet extractor(mir_context_);
		stmt->accept(&extractor);
		for (auto tare: extractor.read_set) {
			generateDeviceToHostCopy(tare);
		}			
		stmt->accept(this);
		for (auto tare: extractor.write_set) {
			generateHostToDeviceCopy(tare);
		}
	}
}
void CodeGenGPU::visit(mir::HybridGPUStmt::Ptr stmt) {
	if (stmt->criteria == fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria::INPUT_VERTEXSET_SIZE) {
		printIndent();
		oss << "if (gpu_runtime::builtin_getVertexSetSize(" << stmt->input_frontier_name << ") < " << stmt->input_frontier_name << ".max_num_elems * ";
		oss << stmt->threshold_var_name;
		oss << ") {" << std::endl;
		indent();
		stmt->stmt1->accept(this);
		dedent();
		printIndent();
		oss << "} else {" << std::endl;
		indent();	
		stmt->stmt2->accept(this);
		dedent();
		printIndent();
		oss << "}" << std::endl;	
	} else {
		assert(false && "Invalid criteria for Hybrid Statement\n");
	}
}
void CodeGenGPUFusedKernel::visit(mir::HybridGPUStmt::Ptr stmt) {
	if (stmt->criteria == fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria::INPUT_VERTEXSET_SIZE) {
		printIndent();
		oss << "if (gpu_runtime::device_builtin_getVertexSetSize(" << var_name(stmt->input_frontier_name) << ") < " << var_name(stmt->input_frontier_name) << ".max_num_elems * ";
		oss << "__device_" << stmt->threshold_var_name;
		oss << ") {" << std::endl;
		indent();
		stmt->stmt1->accept(this);
		dedent();
		printIndent();
		oss << "} else {" << std::endl;
		indent();	
		stmt->stmt2->accept(this);
		dedent();
		printIndent();
		oss << "}" << std::endl;	
	} else {
		assert(false && "Invalid criteria for Hybrid Statement\n");
	}
}

void CodeGenGPU::visit(mir::VertexSetWhereExpr::Ptr expr) {
	assert(false && "VertexSetWhereExpr should be handled in AssignStmt");
}

}
