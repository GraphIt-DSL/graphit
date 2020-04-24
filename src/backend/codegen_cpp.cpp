//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/codegen_cpp.h>
#include <graphit/midend/mir.h>

namespace graphit {
    int CodeGenCPP::genCPP() {
        genIncludeStmts();
        genEdgeSets();
        //genElementData();
        genStructTypeDecls();
        genTypesRequiringTypeDefs();

        //Processing the constants, generting declartions
        for (auto constant : mir_context_->getLoweredConstants()) {
            if ((std::dynamic_pointer_cast<mir::VectorType>(constant->type)) != nullptr) {
                mir::VectorType::Ptr type = std::dynamic_pointer_cast<mir::VectorType>(constant->type);
                // if the constant decl is a field property of an element (system vector)
                //if (type->element_type != nullptr) {
                    //genPropertyArrayImplementationWithInitialization(constant);
                    //NOTE: here we only generate the declaration, not the allocation and initialization
                    // even through we have all the information.
                    // This is because we want to do the allocation and initialization steps in the main function,
                    // when we are using command line arguments and variables. This also allows flexibility of array of structs
                    // and struct of arrays.
                    // To support this feature, we have specialized the code generation of main function (see func_decl visit method).
                    // We first generate allocation, and then initialization (init_stmts) for global variables.
                    genPropertyArrayDecl(constant);
                //}
            } else if (std::dynamic_pointer_cast<mir::VertexSetType>(constant->type)) {
                // if the constant is a vertex set  decl
                // currently, no code is generated
            } else {
                // regular constant declaration
                //constant->accept(this);
                genScalarDecl(constant);
            }
        }

        // Generate global declarations for socket-local buffers used by NUMA optimization
        for (auto iter : mir_context_->edgeset_to_label_to_merge_reduce) {
            for (auto inner_iter : iter.second) {
                if (inner_iter.second->numa_aware) {
                    inner_iter.second->scalar_type->accept(this);
                    oss << " **local_" << inner_iter.second->field_name << ";" << std::endl;
                }
            }
        }

        //Generates function declarations for various edgeset apply operations with different schedules
        // TODO: actually complete the generation, fow now we will use libraries to test a few schedules
        auto gen_edge_apply_function_visitor = EdgesetApplyFunctionDeclGenerator(mir_context_, oss);
        gen_edge_apply_function_visitor.genEdgeApplyFuncDecls();

        //Processing the functions
        std::map<std::string, mir::FuncDecl::Ptr>::iterator it;
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        std::vector<mir::FuncDecl::Ptr> extern_functions = mir_context_->getExternFunctionList();

        for (auto it = extern_functions.begin(); it != extern_functions.end(); it++) {
            it->get()->accept(this);
        }

        for (auto it = functions.begin(); it != functions.end(); it++) {
	        it->get()->accept(this);

        }

	generatePyBindModule();
        oss << std::endl;
        return 0;
    };
    void CodeGenCPP::generatePyBindModule() {
	oss << "#ifdef GEN_PYBIND_WRAPPERS" << std::endl;
	oss << "PYBIND11_MODULE(" << module_name << ", m) {" << std::endl;
	indent();

        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
	for (auto it = functions.begin(); it != functions.end(); it++) {
		mir::FuncDecl::Ptr func_decl = *it;
		if (func_decl->type == mir::FuncDecl::Type::EXPORTED) {
			oss << "m.def(\"" << func_decl->name << "\", &" << func_decl->name << "__wrapper, \"\");" << std::endl;
		}
	}
	dedent();
	oss << "}" << std::endl;		

	oss << "#endif" << std::endl;
    }
    void CodeGenCPP::genIncludeStmts() {
        oss << "#include <iostream> " << std::endl;
        oss << "#include <vector>" << std::endl;
        oss << "#include <algorithm>" << std::endl;
        oss << "#include \"intrinsics.h\"" << std::endl;
	
	oss << "#ifdef GEN_PYBIND_WRAPPERS" << std::endl;
	oss << "#include <pybind11/pybind11.h>" << std::endl;
	oss << "#include <pybind11/stl.h>" << std::endl;
	oss << "#include <pybind11/numpy.h>" << std::endl;
	oss << "namespace py = pybind11;" << std::endl;
	oss << "#endif" << std::endl;
	
	
    }

    void CodeGenCPP::visit(mir::IdentDecl::Ptr ident) {
        oss << ident->name;
    }

    void CodeGenCPP::visit(mir::ForStmt::Ptr for_stmt) {
        printIndent();
        auto for_domain = for_stmt->domain;
        auto loop_var = for_stmt->loopVar;
        oss << "for ( int " << loop_var << " = ";
        for_domain->lower->accept(this);
        oss << "; " << loop_var << " < ";
        for_domain->upper->accept(this);
        oss << "; " << loop_var << "++ )" << std::endl;
        printBeginIndent();
        indent();
        for_stmt->body->accept(this);
        dedent();
        printEndIndent();
        oss << std::endl;

    }

    void CodeGenCPP::visit(mir::WhileStmt::Ptr while_stmt) {
        printIndent();
        oss << "while ( ";
        while_stmt->cond->accept(this);
        oss << ")" << std::endl;
        printBeginIndent();
        indent();
        while_stmt->body->accept(this);
        dedent();
        printEndIndent();
        oss << std::endl;

    }

    void CodeGenCPP::visit(mir::IfStmt::Ptr stmt) {
        printIndent();
        oss << "if (";
        stmt->cond->accept(this);
        oss << ")" << std::endl;

        printIndent();
        oss << " { " << std::endl;

        indent();
        stmt->ifBody->accept(this);
        dedent();

        printIndent();
        oss << " } " << std::endl;

        if (stmt->elseBody) {
            printIndent();
            oss << "else" << std::endl;

            printIndent();
            oss << " { " << std::endl;

            indent();
            stmt->elseBody->accept(this);
            dedent();

            oss << std::endl;

            printIndent();
            oss << " } " << std::endl;

        }

        //printIndent();
        //oss << "end";

    }

    void CodeGenCPP::visit(mir::ExprStmt::Ptr expr_stmt) {
	
        if (mir::isa<mir::EdgeSetApplyExpr>(expr_stmt->expr)) {
	        if (mir::isa<mir::UpdatePriorityEdgeCountEdgeSetApplyExpr>(expr_stmt->expr)) {
                printIndent();
		    expr_stmt->expr->accept(this);
		    oss << ";" << std::endl;
	        } else {
                printIndent();
                auto edgeset_apply_expr = mir::to<mir::EdgeSetApplyExpr>(expr_stmt->expr);
                genEdgesetApplyFunctionCall(edgeset_apply_expr);
	        }
        } else if (mir::isa<mir::PriorityUpdateOperator>(expr_stmt->expr)
                && mir_context_->priority_update_type == mir::PriorityUpdateType::ReduceBeforePriorityUpdate){
            //print an assignment to the tracking variable for PriorityUpdateOperatorSum
            auto priority_update_operator = mir::to<mir::PriorityUpdateOperator>(expr_stmt->expr);
            auto tracking_var = priority_update_operator->tracking_var;
            printIndent();
            oss << tracking_var << " = ";
            priority_update_operator->accept(this);
            oss << ";" << std::endl;
        } else {
            printIndent();
            expr_stmt->expr->accept(this);
            oss << ";" << std::endl;
        }
    }

    void CodeGenCPP::visit(mir::AssignStmt::Ptr assign_stmt) {
        // Removing this special case because the filter is now handled by intrinsics
/*
        if (mir::isa<mir::VertexSetWhereExpr>(assign_stmt->expr)) {
            // declaring a new vertexset as output from where expression
            printIndent();
            assign_stmt->expr->accept(this);
            oss << std::endl;
            printIndent();
            assign_stmt->lhs->accept(this);
            oss << "  = ____graphit_tmp_out; " << std::endl;
        } else 
*/
        if (mir::isa<mir::EdgeSetApplyExpr>(assign_stmt->expr)) {
            printIndent();
            assign_stmt->lhs->accept(this);
            oss << " = ";
            auto edgeset_apply_expr = mir::to<mir::EdgeSetApplyExpr>(assign_stmt->expr);
            genEdgesetApplyFunctionCall(edgeset_apply_expr);

        } else if (mir::isa<mir::EdgeSetLoadExpr>(assign_stmt->expr) &&
                   (mir::to<mir::EdgeSetLoadExpr>(assign_stmt->expr)->priority_update_type ==
                    mir::PriorityUpdateType::ExternPriorityUpdate ||
                    mir::to<mir::EdgeSetLoadExpr>(assign_stmt->expr)->priority_update_type ==
                    mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate)) { // Add other checks here
            printIndent();
            oss << "{" << std::endl;
            indent();
            printIndent();
            assign_stmt->lhs->accept(this);
            oss << " = ";
            //assign_stmt->expr->accept(this);
            auto edgeset_load_expr = mir::to<mir::EdgeSetLoadExpr>(assign_stmt->expr);
            // DO no load the infra_gapbs format 
            /*
	    if (edgeset_load_expr->is_weighted_) {
                oss << "builtin_loadWeightedEdgesFromFile ( ";
                edgeset_load_expr->file_name->accept(this);
		oss << " + std::string(\".wel\")"; 
                oss << ") ";
            } else {
                oss << "builtin_loadEdgesFromFile ( ";
                edgeset_load_expr->file_name->accept(this);
		oss << " + std::string(\".el\")"; 
                oss << ") ";
            }
	
	    oss << ";" << std::endl;
            */
            // Now load the Julienne type graph
            printIndent();
            assign_stmt->lhs->accept(this);
            //oss << ".julienne_graph";
            oss << " = ";
            oss << "builtin_loadJulienneEdgesFromFile(";
            mir::to<mir::EdgeSetLoadExpr>(assign_stmt->expr)->file_name->accept(this);
            oss << ");" << std::endl;

            printIndent();
            assign_stmt->lhs->accept(this);
            oss << ".em = new julienne::EdgeMap<julienne::uintE, julienne::symmetricVertex>(";
            assign_stmt->lhs->accept(this);
            oss << ", std::make_tuple(UINT_E_MAX, 0), (size_t)";
            assign_stmt->lhs->accept(this);
            oss << ".m/5);" << std::endl;

            dedent();
            printIndent();
            oss << "}" << std::endl;

        } else {
            printIndent();
            assign_stmt->lhs->accept(this);
            oss << " = ";
            assign_stmt->expr->accept(this);
            oss << ";" << std::endl;
        }
    }

    void CodeGenCPP::visit(mir::CompareAndSwapStmt::Ptr cas_stmt) {
        printIndent();
        oss << cas_stmt->tracking_var_ << " = compare_and_swap ( ";
        cas_stmt->lhs->accept(this);
        oss << ", ";
        cas_stmt->compare_val_expr->accept(this);
        oss << ", ";
        cas_stmt->expr->accept(this);
        oss << ");" << std::endl;
    }

    void CodeGenCPP::visit(mir::ReduceStmt::Ptr reduce_stmt) {

        if (mir::isa<mir::VertexSetWhereExpr>(reduce_stmt->expr) ||
            mir::isa<mir::EdgeSetApplyExpr>(reduce_stmt->expr)) {


        } else {
            switch (reduce_stmt->reduce_op_) {
                case mir::ReduceStmt::ReductionOp::SUM:
                    printIndent();
                    reduce_stmt->lhs->accept(this);
                    oss << " += ";
                    reduce_stmt->expr->accept(this);
                    oss << ";" << std::endl;

                    if (reduce_stmt->tracking_var_name_ != "") {
                        // need to set the tracking variable
                        printIndent();
                        oss << reduce_stmt->tracking_var_name_ << " = true ; " << std::endl;
                    }

                    break;
                case mir::ReduceStmt::ReductionOp::MIN:
                    printIndent();
                    oss << "if ( ( ";
                    reduce_stmt->lhs->accept(this);
                    oss << ") > ( ";
                    reduce_stmt->expr->accept(this);
                    oss << ") ) { " << std::endl;
                    indent();
                    printIndent();
                    reduce_stmt->lhs->accept(this);
                    oss << "= ";
                    reduce_stmt->expr->accept(this);
                    oss << "; " << std::endl;


                    if (reduce_stmt->tracking_var_name_ != "") {
                        // need to generate a tracking variable
                        printIndent();
                        oss << reduce_stmt->tracking_var_name_ << " = true ; " << std::endl;
                    }

                    dedent();
                    printIndent();
                    oss << "} " << std::endl;
                    break;
                case mir::ReduceStmt::ReductionOp::MAX:
                    //TODO: not supported yet

                    oss << " max= ";
                    break;
                case mir::ReduceStmt::ReductionOp::ATOMIC_MIN:
                    printIndent();
                    oss << reduce_stmt->tracking_var_name_ << " = ";
                    oss << "writeMin( &";
                    reduce_stmt->lhs->accept(this);
                    oss << ", ";
                    reduce_stmt->expr->accept(this);
                    oss << " ); " << std::endl;
                    break;
                case mir::ReduceStmt::ReductionOp::ATOMIC_SUM:
                    printIndent();
                    if (reduce_stmt->tracking_var_name_ != "")
                        oss << reduce_stmt->tracking_var_name_ << " =  true;\n";
                    oss << "writeAdd( &";
                    reduce_stmt->lhs->accept(this);
                    oss << ", ";
                    reduce_stmt->expr->accept(this);
                    oss << " ); " << std::endl;
                    break;
            }

        }
    }

    void CodeGenCPP::visit(mir::PrintStmt::Ptr print_stmt) {
        printIndent();
        oss << "std::cout << ";
        print_stmt->expr->accept(this);
        oss << "<< std::endl;" << std::endl;
    }

    void CodeGenCPP::visit(mir::BreakStmt::Ptr print_stmt) {
        printIndent();
        oss << "break;" << std::endl;
    }

    void CodeGenCPP::visit(mir::VarDecl::Ptr var_decl) {
        // Removing this special case because we want to generate intrinsics for filter
/*
        if (mir::isa<mir::VertexSetWhereExpr>(var_decl->initVal)) {
            // declaring a new vertexset as output from where expression
            printIndent();
            var_decl->initVal->accept(this);
            oss << std::endl;
            printIndent();
            var_decl->type->accept(this);
            oss << var_decl->name << "  = ____graphit_tmp_out; " << std::endl;
        } else 
*/
        if (mir::isa<mir::EdgeSetApplyExpr>(var_decl->initVal)) {
            printIndent();
            var_decl->type->accept(this);
            oss << var_decl->name << " = ";
            auto edgeset_apply_expr = mir::to<mir::EdgeSetApplyExpr>(var_decl->initVal);
            genEdgesetApplyFunctionCall(edgeset_apply_expr);
/*	} else if (mir::isa<mir::VertexSetType>(var_decl->type) && mir::to<mir::VertexSetType>(var_decl->type)->priority_update_type == mir::PriorityUpdateType::ExternPriorityUpdate) {
	    printIndent();
	    oss << "julienne::vertexSubset ";
	    oss << var_decl->name << " ";
	    if (var_decl->initVal != nullptr) {
	        oss << "= ";
	        var_decl->initVal->accept(this);
	    }
	    oss << ";" <<std::endl;
*/
        } else if (mir::isa<mir::VectorType>(var_decl->type)) {

            printIndent();
            genLocalArrayAlloc(var_decl);

        } else {
            printIndent();
            //we probably don't need the modifiers now
            //oss << var_decl->modifier << ' ';
            var_decl->type->accept(this);
            oss << var_decl->name << " ";
            if (var_decl->initVal != nullptr) {
                oss << "= ";
                var_decl->initVal->accept(this);
            }
            oss << ";" << std::endl;

        }
    }

    void CodeGenCPP::visit(mir::FuncExpr::Ptr funcExpr) {

        // if it is extern function, just stop after the name.
        if (mir_context_->isExternFunction(funcExpr->function_name->name)){
            funcExpr->function_name->accept(this);

        } else {
            funcExpr->function_name->accept(this);

            oss << "(";

            bool printDelimiter = false;

            for (auto arg: funcExpr->functorArgs){
                if (printDelimiter) {
                    oss << ",";
                }

                arg->accept(this);
                printDelimiter = true;
            }

            oss << ")";

        }
    }


    void CodeGenCPP::visit(mir::FuncDecl::Ptr func_decl) {

        if (func_decl->type == mir::FuncDecl::Type::EXTERNAL) {
            oss << "extern ";
            if (func_decl->result.isInitialized())
                func_decl->result.getType()->accept(this);
            else
                oss << "void ";
            oss << func_decl->name << " (";

            bool printDelimiter = false;
            for (auto arg : func_decl->args) {
                if (printDelimiter) {
                    oss << ", ";
                }

                arg.getType()->accept(this);
                oss << arg.getName();
                printDelimiter = true;
            }
            if (!printDelimiter)
                oss << "void";
            oss << "); ";
            oss << std::endl;
            return;
        }

        // Generate function signature
        if (func_decl->name == "main") {
            func_decl->isFunctor = false;
            oss << "int " << func_decl->name << "(int argc, char * argv[])";
        } else {
            // Use functors for better compiler inlining
            func_decl->isFunctor = true;
            oss << "struct " << func_decl->name << std::endl;
            printBeginIndent();
            indent();
            //oss << std::string(2 * indentLevel, ' ');

            for (auto arg : func_decl->functorArgs) {

                arg.getType()->accept(this);
                oss << arg.getName() << ";\n";

            }

            bool printDelimiter = false;

            if (!func_decl->functorArgs.empty()){
                oss << func_decl->name;
                oss << "(";

                for (auto arg : func_decl->functorArgs) {
                    if (printDelimiter) {
                        oss << ", ";
                    }

                    arg.getType()->accept(this);
                    oss << arg.getName();
                    printDelimiter = true;
                }
                oss << "): ";

                printDelimiter = false;
                for (auto arg : func_decl->functorArgs) {
                    if (printDelimiter) {
                        oss << ", ";
                    }
                    oss << arg.getName() << "(" << arg.getName() << ")";
                    printDelimiter = true;
                }

                oss << " {}" << std::endl;

            }



            if (func_decl->result.isInitialized()) {
                func_decl->result.getType()->accept(this);

                //insert an additional var_decl for returning result
                const auto var_decl = std::make_shared<mir::VarDecl>();
                var_decl->name = func_decl->result.getName();
                var_decl->type = func_decl->result.getType();
                if (func_decl->body->stmts == nullptr) {
                    func_decl->body->stmts = new std::vector<mir::Stmt::Ptr>();
                }
                auto it = func_decl->body->stmts->begin();
                func_decl->body->stmts->insert(it, var_decl);
            } else {
                oss << "void ";
            }

            oss << "operator() (";

            if (mir_context_->eager_priority_update_edge_function_name == func_decl->name){


                // if this is a priority update edge function for EagerPriorityUpdate with and without merge
                // Then we need to insert an extra argument local bins
                oss << "vector<vector<NodeID>>& local_bins, ";
            }

            printDelimiter = false;

            for (auto arg : func_decl->args) {
                if (printDelimiter) {
                    oss << ", ";
                }

                arg.getType()->accept(this);
                oss << arg.getName();
                printDelimiter = true;
            }
            oss << ") ";
        }

        oss << std::endl;
        printBeginIndent();
        indent();

        if (func_decl->name == "main") {
            //generate special initialization code for main function
            //TODO: this is probably a hack that could be fixed for later

            //First, allocate the edgesets (read them from outside files if needed)
            for (auto stmt : mir_context_->edgeset_alloc_stmts) {
                stmt->accept(this);
            }

            // Initialize graphSegments if necessary
            auto segment_map = mir_context_->edgeset_to_label_to_num_segment;
            for (auto edge_iter = segment_map.begin(); edge_iter != segment_map.end(); edge_iter++) {
                auto edgeset = mir_context_->getConstEdgeSetByName((*edge_iter).first);
                auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
                bool is_weighted = (edge_set_type->weight_type != nullptr);
                for (auto label_iter = (*edge_iter).second.begin();
                     label_iter != (*edge_iter).second.end(); label_iter++) {
                    auto edge_iter_first = (*edge_iter).first;
                    auto label_iter_first = (*label_iter).first;
                    auto label_iter_second = (*label_iter).second;
                    auto numa_aware_flag = mir_context_->edgeset_to_label_to_merge_reduce[edge_iter_first][label_iter_first]->numa_aware;

                    if (label_iter_second < 0) {
                        //do a specical case for negative number of segments. I
                        // in the case of negative integer, we use the number as argument to runtimve argument argv
                        // this is the only place in the generated code that we set the number of segments
                        oss << "  " << edgeset->name << ".buildPullSegmentedGraphs(\"" << label_iter_first
                            << "\", " << "atoi(argv[" << -1 * label_iter_second << "])"
                            << (numa_aware_flag ? ", true" : "") << ");" << std::endl;
                    } else {
                        // just use the positive integer as argument to number of segments
                        oss << "  " << edgeset->name << ".buildPullSegmentedGraphs(\"" << label_iter_first
                            << "\", " << label_iter_second
                            << (numa_aware_flag ? ", true" : "") << ");" << std::endl;
                    }
                }
            }

            //generate allocation statemetns for field vectors
            for (auto constant : mir_context_->getLoweredConstants()) {
                if ((std::dynamic_pointer_cast<mir::VectorType>(constant->type)) != nullptr) {
                    mir::VectorType::Ptr type = std::dynamic_pointer_cast<mir::VectorType>(constant->type);
                    // if the constant decl is a field property of an element (system vector)
                    if (type->element_type != nullptr) {
                        //genPropertyArrayImplementationWithInitialization(constant);
                        //genPropertyArrayDecl(constant);
                        if (constant->needs_allocation)
                            genPropertyArrayAlloc(constant);
                    } else {
                        //constant scalar vector
                        genScalarVectorAlloc(constant, type);
                    }
                } else if (std::dynamic_pointer_cast<mir::VertexSetType>(constant->type) ||
                        std::dynamic_pointer_cast<mir::PriorityQueueType>(constant->type)){
                    // if the constant is a vertex set  decl
                    //if it is a priorty queue type, then don't do anything

                }else {
                    // regular constant declaration
                    //constant->accept(this);
                    if(constant->initVal != nullptr){
                        genScalarAlloc(constant);
                    }
                }
            }

            // the stmts that initializes the field vectors
            for (auto stmt : mir_context_->field_vector_init_stmts) {
                stmt->accept(this);
            }

            for (auto iter : mir_context_->edgeset_to_label_to_merge_reduce) {
                for (auto inner_iter : iter.second) {

                    if ((inner_iter.second)->numa_aware) {
                        auto merge_reduce = inner_iter.second;
                        std::string local_field = "local_" + merge_reduce->field_name;
                        oss << "  " << local_field << " = new ";
                        merge_reduce->scalar_type->accept(this);
                        oss << "*[omp_get_num_places()];\n";

                        oss << "  for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {\n";
                        oss << "    " << local_field << "[socketId] = (";
                        merge_reduce->scalar_type->accept(this);
                        oss << "*)numa_alloc_onnode(sizeof(";
                        merge_reduce->scalar_type->accept(this);
                        oss << ") * ";
                        auto count_expr = mir_context_->getElementCount(
                                mir_context_->getElementTypeFromVectorOrSetName(merge_reduce->field_name));
                        count_expr->accept(this);
                        oss << ", socketId);\n";

                        oss << "    ligra::parallel_for_lambda((int)0, (int)";
                        count_expr->accept(this);
                        oss << ", [&] (int n) {\n";
                        oss << "      " << local_field << "[socketId][n] = " << merge_reduce->field_name << "[n];\n";
                        oss << "    });\n  }\n";

                        oss << "  omp_set_nested(1);" << std::endl;
                    }
                }
            }
        } //end of if "main" condition

        // still generate the constant declarations
        if (func_decl->type == mir::FuncDecl::Type::EXPORTED){
            for (auto constant : mir_context_->getLoweredConstants()) {
                if (mir::isa<mir::ScalarType>(constant->type) &&
                        constant->initVal != nullptr){
                    genScalarAlloc(constant);
                }
            }
        }


        //if the function has a body
        if (func_decl->body && func_decl->body->stmts) {


            func_decl->body->accept(this);

            //print a return statemetn if there is a result
            if (func_decl->result.isInitialized()) {
                printIndent();
                oss << "return " << func_decl->result.getName() << ";" << std::endl;
            }


        }

        if (func_decl->isFunctor) {
            dedent();
            printEndIndent();
            oss << ";";
            oss << std::endl;
        }

        if (func_decl->name == "main") {
            for (auto iter : mir_context_->edgeset_to_label_to_merge_reduce) {
                for (auto inner_iter : iter.second) {
                    if (inner_iter.second->numa_aware) {
                        auto merge_reduce = inner_iter.second;
                        oss << "  for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {\n";
                        oss << "    numa_free(local_" << merge_reduce->field_name << "[socketId], sizeof(";
                        merge_reduce->scalar_type->accept(this);
                        oss << ") * ";
                        mir_context_->getElementCount(
                                mir_context_->getElementTypeFromVectorOrSetName(merge_reduce->field_name))->accept(
                                this);
                        oss << ");\n  }\n";
                    }
                }
            }
        }

        dedent();
        printEndIndent();
        oss << ";";
        oss << std::endl;
	if (func_decl-> type == mir::FuncDecl::Type::EXPORTED) {
		generatePyBindWrapper(func_decl);
	}
    };
    void CodeGenCPP::generatePyBindWrapper(mir::FuncDecl::Ptr func_decl) {
	    oss << "#ifdef GEN_PYBIND_WRAPPERS" << std::endl;
	    oss << "//PyBind Wrappers for function" << func_decl->name << std::endl;
	    //Currently we do no support, returning Graph Types. So return type can be directly emitted without extra checks	
	    if (func_decl->result.isInitialized())
		    if (mir::isa<mir::VectorType>(func_decl->result.getType())) {

		        mir::VectorType::Ptr vector_type = mir::to<mir::VectorType>(func_decl->result.getType());
			oss << "py::array_t<";

			if (mir::isa<mir::VectorType>(vector_type->vector_element_type)) 
				mir::to<mir::VectorType>(vector_type->vector_element_type)->vector_element_type->accept(this);
			else 
				vector_type->vector_element_type->accept(this);
			oss << "> ";
		    }
		    else 
		        func_decl->result.getType()->accept(this);
	    else
		    oss << "void ";
	    oss << func_decl->name << "__wrapper (";
	    // For argument types we need to check if it is a graph, if it is, we need to expand it into 3 numpy arrays
	    bool printDelimiter = false;
	    for (auto arg : func_decl->args) {
		    if (printDelimiter) {
			    oss << ", ";
		    }
		    if (mir::isa<mir::EdgeSetType>(arg.getType())) {
			    oss << "py::object _" << arg.getName();
		    }else if (mir::isa<mir::VectorType>(arg.getType())) {
			    // We want to support vectors of vectors of scalar types separately
		            mir::VectorType::Ptr vector_type = mir::to<mir::VectorType>(arg.getType());
			    mir::Type::Ptr elem_type = vector_type->vector_element_type;
			    if (mir::isa<mir::VectorType>(elem_type)) {
				    mir::VectorType::Ptr inner_vector_type = mir::to<mir::VectorType>(elem_type);
				    oss << "py::array_t<";
                                    inner_vector_type->vector_element_type->accept(this);
                                    oss << ">";
                                    oss << " _" << arg.getName(); 
			    }else { 
				    oss << "py::array_t<";
				    vector_type->vector_element_type->accept(this);
				    oss << ">";	
				    oss << " _" << arg.getName();
			    }
		    }else {
			    arg.getType()->accept(this);
			    oss << arg.getName();
		    }
		    printDelimiter = true;
	    }
	    if (!printDelimiter)
		    oss << "void";
	    oss << ") ";	
	    oss << "{ " << std::endl;
	    indent();
	    // Need to generate translation for graph arguments before the actual call

	    for (auto arg : func_decl->args) {
		    if (mir::isa<mir::EdgeSetType>(arg.getType())) {
			    mir::EdgeSetType::Ptr type = mir::to<mir::EdgeSetType>(arg.getType());
			    if (type->weight_type != NULL) {
				   printIndent();
				   oss << "py::array_t<";
				   type->weight_type->accept(this);
				   oss << "> " << arg.getName() << "__data = _" << arg.getName() << ".attr(\"data\").cast<py::array_t<"; 
				   type->weight_type->accept(this);
				   oss << ">>();" << std::endl;

				    printIndent();
				    oss << "py::array_t<int> " << arg.getName() << "__indices = _" << arg.getName() << ".attr(\"indices\").cast<py::array_t<int>>();" << std::endl;
				    printIndent();
				    oss << "py::array_t<int> " << arg.getName() << "__indptr = _" << arg.getName() << ".attr(\"indptr\").cast<py::array_t<int>>();" << std::endl;
				    printIndent();
				    arg.getType()->accept(this);
				    oss << arg.getName() << " = builtin_loadWeightedEdgesFromCSR(";
				    oss << arg.getName() << "__data.data(), " << arg.getName() << "__indptr.data(), " << arg.getName() << "__indices.data(), " << arg.getName() << "__indptr.size()-1, " << arg.getName() << "__indices.size());" << std::endl; 
			    } else {	
				    //Prepare the individual arrays from the object
				    printIndent();
				    oss << "py::array_t<int> " << arg.getName() << "__data = _" << arg.getName() << ".attr(\"data\").cast<py::array_t<int>>();" << std::endl;
				    printIndent();
				    oss << "py::array_t<int> " << arg.getName() << "__indices = _" << arg.getName() << ".attr(\"indices\").cast<py::array_t<int>>();" << std::endl;
				    printIndent();
				    oss << "py::array_t<int> " << arg.getName() << "__indptr = _" << arg.getName() << ".attr(\"indptr\").cast<py::array_t<int>>();" << std::endl;
				    printIndent();
				    arg.getType()->accept(this);
				    oss << arg.getName() << " = builtin_loadEdgesFromCSR(";
				    oss << arg.getName() << "__indptr.data(), " << arg.getName() << "__indices.data(), " << arg.getName() << "__indptr.size()-1, " << arg.getName() << "__indices.size());" << std::endl; 
			    }
			
		    } else if (mir::isa<mir::VectorType>(arg.getType())) {
			    mir::VectorType::Ptr vector_type = mir::to<mir::VectorType>(arg.getType());
			    printIndent();
			    vector_type->accept(this);
			    oss << " " << arg.getName() << " = (";
			    vector_type->accept(this);
			    oss << ")_" << arg.getName() << ".data();" << std::endl; 
		    }

	    }
	    printIndent();
		
	    if (func_decl->result.isInitialized()) {
		    
		    func_decl->result.getType()->accept(this);
		    oss << "__" << func_decl->result.getName() << " = ";
	    }
	    oss << func_decl->name;
	    if (func_decl->isFunctor)
		    oss << "()";
	    oss << "(";
	    printDelimiter = false;
	    for (auto arg : func_decl->args) {
		    if (printDelimiter) {
			    oss << ", ";
		    }
		    oss << arg.getName();
		    printDelimiter = true;
	    }
	    oss << ");" << std::endl;
	    // We do no support returning Graph types. But we can return still return vectors	    
	    if (func_decl->result.isInitialized() ) { 
		    if (mir::isa<mir::VectorType>(func_decl->result.getType())) {
			    mir::VectorType::Ptr vector_type = mir::to<mir::VectorType>(func_decl->result.getType());
			    // Handle separately if vector of vector
			    if (mir::isa<mir::VectorType>(vector_type->vector_element_type)) {
				    mir::VectorType::Ptr inner_vector_type = mir::to<mir::VectorType> (vector_type->vector_element_type);
				    printIndent();
				    oss << "py::array_t<";
				    inner_vector_type->vector_element_type->accept(this);
				    oss << "> " << func_decl->result.getName() << " = py::array_t<";
				    inner_vector_type->vector_element_type->accept(this);
				    oss << "> ( std::vector<size_t>{(size_t)";
				    if (vector_type->element_type != nullptr)
				        mir_context_->getElementCount(vector_type->element_type)->accept(this);
				    else
					oss << vector_type->range_indexset;
				    oss << ", (size_t)";
				    oss << inner_vector_type->range_indexset;
				    oss << "}, std::vector<size_t>{ ";
				    oss << "( " << inner_vector_type->range_indexset << " * " << "sizeof(";
				    inner_vector_type->vector_element_type->accept(this);
				    oss << ")), sizeof(";
				    inner_vector_type->vector_element_type->accept(this);
				    oss << ") }, (";
				    inner_vector_type->vector_element_type->accept(this);
				    oss << "*)__" << func_decl->result.getName() << ");" << std::endl;
				
				    
			    } else   {
				    // Create the return object
				    printIndent();
				    oss << "py::array_t<";
				    vector_type->vector_element_type->accept(this);
				    oss << "> " << func_decl->result.getName() << " = py::array_t<";
				    vector_type->vector_element_type->accept(this);
				    oss << "> ( {";
				    if (vector_type->element_type != nullptr){
				        // get the size information of the output by looking up the count of the associated Element (e.g. Vertex) type
                        mir_context_->getElementCount(vector_type->element_type)->accept(this);
				    } else if (vector_type->range_indexset > 0) {
				        // the vector has range index associated with it
				        oss << vector_type->range_indexset;
				    }
				    oss << "}, { sizeof(";
				    vector_type->vector_element_type->accept(this);
				    oss << ") }, __" << func_decl->result.getName() << ");" << std::endl; 
			    }
			    
		    } else {
		            printIndent();
			    func_decl->result.getType()->accept(this);
			    oss << func_decl->result.getName() << " = __";
			    oss << func_decl->result.getName() << ";" << std::endl;
		    }
	    }
	    if (func_decl->result.isInitialized()) {
		    printIndent();
		    oss << "return " << func_decl->result.getName() << ";" << std::endl;
	    }		
	    dedent();
	    printIndent();
	    oss << "}" << std::endl;
	    oss << "#endif" << std::endl;

    }

    void CodeGenCPP::visit(mir::ScalarType::Ptr scalar_type) {
        switch (scalar_type->type) {
            case mir::ScalarType::Type::INT:
                oss << "int ";
                break;
            case mir::ScalarType::Type::UINT:
                oss << "uintE ";
                break;
            case mir::ScalarType::Type::UINT_64:
                oss << "uint64_t ";
                break;
            case mir::ScalarType::Type::FLOAT:
                oss << "float ";
                break;
            case mir::ScalarType::Type::DOUBLE:
                oss << "double ";
                break;
            case mir::ScalarType::Type::BOOL:
                oss << "bool ";
                break;
            case mir::ScalarType::Type::STRING:
                oss << "string ";
                break;
            default:
                break;
        }
    }

    void CodeGenCPP::visit(mir::VectorType::Ptr vector_type) {
        if (mir::isa<mir::ScalarType>(vector_type->vector_element_type)){
            vector_type->vector_element_type->accept(this);
        } else if (mir::isa<mir::VectorType>(vector_type->vector_element_type)){
            //nested vector type
            mir::VectorType::Ptr inner_vector_type = mir::to<mir::VectorType>(vector_type->vector_element_type);
            // use the typedef type for the inner vector type
            oss << inner_vector_type->toString();
        }
        oss << " * ";
    }

    void CodeGenCPP::visit(mir::StructTypeDecl::Ptr struct_type) {
        oss << struct_type->name << " ";
    }

    void CodeGenCPP::visit(mir::Call::Ptr call_expr) {

        bool call_on_built_in_priority_queue = false;
        std::string priority_queue_name = "";

        //check if this is a call on priority queue
        if (mir_context_->getPriorityQueueDecl() != nullptr){
            if (call_expr->args.size() > 0
                && mir::isa<mir::VarExpr>(call_expr->args[0])) {
                auto target_name_expr = mir::to<mir::VarExpr>(call_expr->args[0]);
                auto target_name = target_name_expr->var.getName();
                priority_queue_name = mir_context_->getPriorityQueueDecl()->name;
                if (target_name == priority_queue_name){
                    call_on_built_in_priority_queue = true;
                }
            }
        }
        // one exception is for dequeue_ready_set for ReduceBeforeUpdate schedule, we do not make the reformat
        if (call_expr->name == "getBucketWithGraphItVertexSubset" && mir_context_->priority_update_type == mir::ReduceBeforePriorityUpdate){
            call_on_built_in_priority_queue = false;
        }

        if (call_on_built_in_priority_queue){
            oss << priority_queue_name << "->";
        }
        oss << call_expr->name;


        if (call_expr->generic_type != nullptr) {
            oss << " < ";
            call_expr->generic_type->accept(this);
            oss << " > ";
        }

        if (mir_context_->isFunction(call_expr->name)) {
            auto mir_func_decl = mir_context_->getFunction(call_expr->name);
            if (mir_func_decl->isFunctor) {
                oss << "(";
                bool printDelimiter = false;

                for (auto arg : call_expr->functorArgs) {
                    if (printDelimiter) {
                        oss << ", ";
                    }
                    arg->accept(this);
                    printDelimiter = true;
                }

                oss << ")";
            }
        }




        oss << "(";


        if (!call_on_built_in_priority_queue){
            bool printDelimiter = false;

            for (auto arg : call_expr->args) {
                if (printDelimiter) {
                    oss << ", ";
                }
                arg->accept(this);
                printDelimiter = true;
            }

            oss << ") ";
        } else {
            //ignore the first argument if it is working on a priority queue
            // generate pq.finished() style code for priority queue

            bool printDelimiter = false;

            for (int i = 1; i < call_expr->args.size(); i++){
                auto arg = call_expr->args[i];
                if (printDelimiter) {
                    oss << ", ";
                }
                arg->accept(this);
                printDelimiter = true;
            }
	    oss << ") ";
        }

    };

    /**
     * DEPRECATED, we don't generate code for TensorReadExpr
     * We only generate code for TensorArrayReadExpr and TensorStructReadExpr
     * @param expr
     */
//    void CodeGenCPP::visit(mir::TensorReadExpr::Ptr expr) {
//        //for dense array tensor read
//        expr->target->accept(this);
//        oss << "[";
//        expr->index->accept(this);
//        oss << "]";
//    };

/**
 * Generate tensor read code for array implementation
 * @param expr
 */
    void CodeGenCPP::visit(mir::TensorArrayReadExpr::Ptr expr) {
        //for dense array tensor read
//        expr->target->accept(this);
//        oss << "[";
//        expr->index->accept(this);
//        oss << "]";
//    }
        if (mir::isa<mir::MIRNode>(expr.get()->target.get()->shared_from_this())) {
            //not sure what this is std::shared_ptr<mir::MIRNode> ptr = expr.get()->target.get()->shared_from_this();
            std::string nameptr = expr.get()->getTargetNameStr();
            if (nameptr == "argv"){
                expr->target->accept(this);
                oss << "_safe(";
                expr->index->accept(this);
                oss << ", argv, argc)";
            }
            else{
                expr->target->accept(this);
                oss << "[";
                expr->index->accept(this);
                oss << "]";
            }

        } else {
            expr->target->accept(this);
            oss << "[";
            expr->index->accept(this);
            oss << "]";
        }
    }

/**
 * Generate tensor read code for struct implementation
 * @param expr
 */
    void CodeGenCPP::visit(mir::TensorStructReadExpr::Ptr expr) {
        //for dense array tensor read
        oss << expr->array_of_struct_target << "[";
        expr->index->accept(this);
        oss << "].";
        expr->field_target->accept(this);
        oss << " ";
    };


    void CodeGenCPP::visit(mir::VarExpr::Ptr expr) {
        if (mir::isa<mir::EdgeSetType>(expr->var.getType())) {  
            mir::PriorityUpdateType update_type = mir::to<mir::EdgeSetType>(expr->var.getType())->priority_update_type;
            if (update_type == mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate || update_type == mir::PriorityUpdateType::ExternPriorityUpdate) {
                oss << expr->var.getName() << ".julienne_graph";
                return;
            }

        }
           
        oss << expr->var.getName();
    };

    void CodeGenCPP::visit(mir::EqExpr::Ptr expr) {
        oss << "(";
        expr->operands[0]->accept(this);
        oss << ")";

        for (unsigned i = 0; i < expr->ops.size(); ++i) {
            switch (expr->ops[i]) {
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
                    break;
            }

            oss << "(";
            expr->operands[i + 1]->accept(this);
            oss << ")";
        }
    }

    void CodeGenCPP::visit(mir::AndExpr::Ptr expr) {
        oss << '(';
        expr->lhs->accept(this);
        oss << " && ";
        expr->rhs->accept(this);
        oss << ')';
    };

    void CodeGenCPP::visit(mir::OrExpr::Ptr expr) {
        oss << '(';
        expr->lhs->accept(this);
        oss << " || ";
        expr->rhs->accept(this);
        oss << ')';
    };

    void CodeGenCPP::visit(mir::XorExpr::Ptr expr) {
        oss << '(';
        expr->lhs->accept(this);
        oss << " ^ ";
        expr->rhs->accept(this);
        oss << ')';
    };

    void CodeGenCPP::visit(mir::NotExpr::Ptr not_expr) {
        oss << " !(";
        not_expr->operand->accept(this);
        oss << ')';
    }

    void CodeGenCPP::visit(mir::MulExpr::Ptr expr) {
        oss << '(';
        expr->lhs->accept(this);
        oss << " * ";
        expr->rhs->accept(this);
        oss << ')';
    }

    void CodeGenCPP::visit(mir::DivExpr::Ptr expr) {
        oss << '(';
        expr->lhs->accept(this);
        oss << " / ";
        expr->rhs->accept(this);
        oss << ')';
    }

    void CodeGenCPP::visit(mir::AddExpr::Ptr expr) {
        oss << '(';
        expr->lhs->accept(this);
        oss << " + ";
        expr->rhs->accept(this);
        oss << ')';
    };

    void CodeGenCPP::visit(mir::SubExpr::Ptr expr) {
        oss << '(';
        expr->lhs->accept(this);
        oss << " - ";
        expr->rhs->accept(this);
        oss << ')';
    };

    void CodeGenCPP::visit(mir::BoolLiteral::Ptr expr) {
        oss << "(bool) ";
        oss << (bool) expr->val;
    };

    void CodeGenCPP::visit(mir::StringLiteral::Ptr expr) {
        oss << "\"";
        oss << expr->val;
        oss << "\"";
    };

    void CodeGenCPP::visit(mir::FloatLiteral::Ptr expr) {
        oss << "(";
        oss << "(float) ";
        oss << expr->val;
        oss << ") ";
    };

    void CodeGenCPP::visit(mir::IntLiteral::Ptr expr) {
        oss << "(";
        //oss << "(int) ";
        oss << expr->val;
        oss << ") ";
    }

    // Materialize the Element Data
    void CodeGenCPP::genElementData() {
        for (auto const &element_type_entry : mir_context_->properties_map_) {
            // for each element type
            for (auto const &var_decl : *element_type_entry.second) {
                // for each field / system vector of the element
                // generate just array implementation for now
                genPropertyArrayImplementationWithInitialization(var_decl);
            }
        }

    };

    void CodeGenCPP::genScalarDecl(mir::VarDecl::Ptr var_decl) {
        //the declaration and the value are separate. The value is generated as a separate assign statement in the main function
        var_decl->type->accept(this);
        oss << var_decl->name << "; " << std::endl;
    }


    void CodeGenCPP::genScalarAlloc(mir::VarDecl::Ptr var_decl) {

        printIndent();

        oss << var_decl->name << " ";
        if (var_decl->initVal != nullptr) {
            oss << "= ";
            var_decl->initVal->accept(this);
        }
        oss << ";" << std::endl;

    }

    void CodeGenCPP::genPropertyArrayDecl(mir::VarDecl::Ptr var_decl) {
        // read the name of the array
        const auto name = var_decl->name;

        // read the type of the array
        mir::VectorType::Ptr vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
        assert(vector_type != nullptr);
        auto vector_element_type = vector_type->vector_element_type;
        assert(vector_element_type != nullptr);

        /**  Deprecated, now we generate an array declaration, not a vector one
        //generate std::vector implementation
        oss << "std::vector< ";
        vector_element_type->accept(this);
        // pointer declaration
        oss << " >  ";
        oss << name;
        oss << ";" << std::endl;
         **/

        if (!mir::isa<mir::VectorType>(vector_element_type)) {
            vector_element_type->accept(this);
            oss << " * __restrict " << name << ";" << std::endl;
        } else if (mir::isa<mir::VectorType>(vector_element_type)) {
            //if each element is a vector
            auto vector_vector_element_type = mir::to<mir::VectorType>(vector_element_type);
            assert(vector_vector_element_type->range_indexset != 0);
            int range = vector_vector_element_type->range_indexset;



            //std::string typedef_name = "defined_type_" + mir_context_->getUniqueNameCounterString();
            std::string typedef_name = vector_vector_element_type->toString();
            if (mir_context_->defined_types.find(typedef_name) == mir_context_->defined_types.end()){
                mir_context_->defined_types.insert(typedef_name);
                //first generates a typedef for the vector type
                oss << "typedef ";
                vector_vector_element_type->vector_element_type->accept(this);
                oss << typedef_name <<  " ";
                oss << "[ " << range << "]; " << std::endl;
            }

            vector_vector_element_type->typedef_name_ = typedef_name;


            //use the typedef defined type to declare a new pointer
            oss << typedef_name << " * __restrict  " << name << ";" << std::endl;

        } else {
            std::cout << "unsupported type for property: " << var_decl->name << std::endl;
            exit(0);
        }
    }

    void CodeGenCPP::genLocalArrayAlloc(mir::VarDecl::Ptr var_decl) {
        // read the name of the array
        const auto name = var_decl->name;

        // read the type of the array
        mir::VectorType::Ptr vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
        assert(vector_type != nullptr);


        vector_type->accept(this);
        oss << name << " ";


        const auto init_val = var_decl->initVal;

        if(init_val != nullptr) {

            if (std::dynamic_pointer_cast<mir::Call>(init_val)) {
                auto call_expr = std::dynamic_pointer_cast<mir::Call>(init_val);
                oss << " = ";
                call_expr->accept(this);
                oss << ";" << std::endl;

            } else if (isLiteral(init_val)){
                oss << " = new ";
                const auto vector_element_type = vector_type->vector_element_type;
                vector_element_type->accept(this);
                const auto size_expr = mir_context_->getElementCount(vector_type->element_type);
                oss << " [ ";
                size_expr->accept(this);
                oss << " ]; " << std::endl;
                printIndent();
                oss << "ligra::parallel_for_lambda((int)0, (int)";
                size_expr->accept(this);
                oss << ", [&] (int i) { ";
                oss << name << "[i]=";
                init_val->accept(this);
                oss << "; });" << std::endl;

            } else {
                oss << " = ";
                var_decl->initVal->accept(this);
                oss << ";" << std::endl;

            }

        }
        else {
            oss << ";" << std::endl;
        }

    }

    void CodeGenCPP::genPropertyArrayAlloc(mir::VarDecl::Ptr var_decl) {
        const auto name = var_decl->name;
        printIndent();
        oss << name;
        // read the size of the array
        mir::VectorType::Ptr vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
        const auto size_expr = mir_context_->getElementCount(vector_type->element_type);
        auto vector_element_type = vector_type->vector_element_type;

        assert(size_expr != nullptr);

        /** Deprecated, now we uses a "new" allocation scheme for arrays
        oss << " = std::vector< ";
        vector_element_type->accept(this);
        oss << " >  ";
        oss << " ( ";
        size_expr->accept(this);
        oss << " ); " << std::endl;
         **/

        oss << " = new ";

        if (mir::isa<mir::VectorType>(vector_element_type)) {
            //for vector type, we use the name from typedef
            auto vector_type_vector_element_type = mir::to<mir::VectorType>(vector_element_type);
            assert(vector_type_vector_element_type->typedef_name_ != "");
            oss << vector_type_vector_element_type->typedef_name_ << " ";
        } else {
            vector_element_type->accept(this);
        }

        oss << "[ ";
        size_expr->accept(this);
        oss << "];" << std::endl;
    }

    void CodeGenCPP::genPropertyArrayImplementationWithInitialization(mir::VarDecl::Ptr var_decl) {
        // read the name of the array
        const auto name = var_decl->name;

        // read the type of the array
        mir::VectorType::Ptr vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
        assert(vector_type != nullptr);
        auto vector_element_type = vector_type->vector_element_type;
        assert(vector_element_type != nullptr);

        //generate std::vector implementation
        oss << "std::vector< ";
        vector_element_type->accept(this);
        // pointer declaration
        oss << " >  ";
        oss << name;

        // read the size of the array
        const auto size_expr = mir_context_->getElementCount(vector_type->element_type);
        assert(size_expr != nullptr);
        const auto init_val = var_decl->initVal;

        if (std::dynamic_pointer_cast<mir::Call>(init_val)) {
            auto call_expr = std::dynamic_pointer_cast<mir::Call>(init_val);
            oss << " = ";
            call_expr->accept(this);
            oss << ";" << std::endl;

        } else {
            oss << " ( ";
            size_expr->accept(this);
            if (init_val) {
                // struct types don't have initial values
                oss << " , ";
                init_val->accept(this);
            }
            oss << " ); " << std::endl;

        }
    }

    void CodeGenCPP::visit(mir::ElementType::Ptr element_type) {
        //currently, we generate an index id into the vectors
        oss << "NodeID ";
    }

    void CodeGenCPP::visit(mir::VertexSetAllocExpr::Ptr alloc_expr) {
        oss << "new VertexSubset<int> ( ";
        //This is the current number of elements, but we need the range
        //alloc_expr->size_expr->accept(this);
        const auto size_expr = mir_context_->getElementCount(alloc_expr->element_type);
        size_expr->accept(this);
        oss << " , ";
        alloc_expr->size_expr->accept(this);
        oss << ")";
    }

    void CodeGenCPP::visit(mir::ListAllocExpr::Ptr alloc_expr) {
        oss << "new std::vector< ";
        alloc_expr->element_type->accept(this);
        oss << " > ( ";
        // currently we don't support initializing a vector with size
        //This is the current number of elements, but we need the range
        //alloc_expr->size_expr->accept(this);
        //const auto size_expr = mir_context_->getElementCount(alloc_expr->element_type);
        //size_expr->accept(this);
        //oss << " , ";
        //alloc_expr->size_expr->accept(this);
        oss << ")";
    }


    void CodeGenCPP::visit(mir::VertexSetApplyExpr::Ptr apply_expr) {
        //vertexset apply
        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(apply_expr->target);

        if (mir_context_->isConstVertexSet(mir_var->var.getName())) {
            //if the verstexset is a const / global vertexset, then we can get size easily
            auto associated_element_type = mir_context_->getElementTypeFromVectorOrSetName(mir_var->var.getName());
            assert(associated_element_type);
            auto associated_element_type_size = mir_context_->getElementCount(associated_element_type);
            assert(associated_element_type_size);
            if (apply_expr->is_parallel) {
                oss << "ligra::parallel_for_lambda((int)0, (int)";
                associated_element_type_size->accept(this);
                oss << ", [&] (int vertexsetapply_iter) {" << std::endl;
            } else {
                oss << "for" << " (int vertexsetapply_iter = 0; vertexsetapply_iter < ";
                associated_element_type_size->accept(this);
                oss << "; vertexsetapply_iter++) {" << std::endl;
            }
            indent();
            printIndent();

            // if functor arg is not empty, we wrap another paranthesis to not confuse it with vertexsetapply_iter
            if (!apply_expr->input_function->functorArgs.empty()) {
                oss << "(";
                apply_expr->input_function->accept(this);
                oss << ")(vertexsetapply_iter);" << std::endl;
            } else {
                apply_expr->input_function->accept(this);
                oss << "(vertexsetapply_iter);" << std::endl;
            }


            dedent();
            printIndent();
            if (apply_expr->is_parallel) {
                oss << "});";
            } else {
                oss << "}";
            }
        } else {
            // NOT sure what how this condition is triggered and used
            // if this is a dynamically created vertexset
            oss << " builtin_vertexset_apply ( " << mir_var->var.getName() << ", ";
            apply_expr->input_function->accept(this);
            oss << " );" << std::endl;


        }


    }

    void CodeGenCPP::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {

        // get the name of the function declaration
        //edgeset apply
        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(apply_expr->target);
        // DEPRECATED, we now do it at various statements (assign)
        // TODO: fix using this visitor again, so we don't need to do it at statements
        //generate code for pull edgeset apply
        //genEdgeSetPullApply(mir_var, apply_expr);
    }

    void CodeGenCPP::visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr) {
        // generate code for push edgeset apply
        // DEPRECATED, we now do it at various statements (assign)
        // TODO: fix using this visitor again, so we don't need to do it at statements
        //genEdgesetPushApply(apply_expr);
    }

    void CodeGenCPP::visit(mir::VertexSetWhereExpr::Ptr vertexset_where_expr) {

	// Removing all this code to just generate calls to builtin
        //dense vertex set apply
/*
        if (vertexset_where_expr->is_constant_set) {
            auto associated_element_type =
                    mir_context_->getElementTypeFromVectorOrSetName(vertexset_where_expr->target);
            assert(associated_element_type);
            auto associated_element_type_size = mir_context_->getElementCount(associated_element_type);
            assert(associated_element_type_size);
            oss << "auto ____graphit_tmp_out = new VertexSubset <NodeID> ( ";
            //get the total number of vertices in the vertex set
            auto vertex_type = mir_context_->getElementTypeFromVectorOrSetName(vertexset_where_expr->target);
            auto vertices_range_expr =
                    mir_context_->getElementCount(vertex_type);
            vertices_range_expr->accept(this);
            oss << " , ";
            //vertices_range_expr->accept(this);
            // the output vertexset is initially set to 0
            oss << "0";
            oss << " );" << std::endl;
            std::string next_bool_map_name = "next" + mir_context_->getUniqueNameCounterString();
            oss << "bool * " << next_bool_map_name << " = newA(bool, ";
            vertices_range_expr->accept(this);
            oss << ");\n";
            printIndent();
            oss << "parallel_for (int v = 0; v < ";
            associated_element_type_size->accept(this);
            oss << "; v++) {" << std::endl;
            indent();
            printIndent();
            oss << next_bool_map_name << "[v] = 0;" << std::endl;
            oss << "if ( " << vertexset_where_expr->input_func << "()( v ) )" << std::endl;
            indent();
            printIndent();
            oss << next_bool_map_name << "[v] = 1;" << std::endl;
            dedent();
            dedent();
            printIndent();
            oss << "} //end of loop\n";
            oss << "____graphit_tmp_out->num_vertices_ = sequence::sum( "
                << next_bool_map_name << ", ";
            vertices_range_expr->accept(this);
            oss << " );\n"
                   "____graphit_tmp_out->bool_map_ = ";
            oss << next_bool_map_name << ";\n";
        }
*/
        if (vertexset_where_expr->is_constant_set) {

            auto associated_element_type =
                    mir_context_->getElementTypeFromVectorOrSetName(vertexset_where_expr->target);
            auto associated_element_type_size = mir_context_->getElementCount(associated_element_type);
            oss << "builtin_const_vertexset_filter <";
	        vertexset_where_expr->input_func->function_name->accept(this);
            oss << ">(";
            vertexset_where_expr->input_func->accept(this);
            oss << ", ";
            associated_element_type_size->accept(this);
            oss << ")";
        } else {
            oss << "builtin_vertexset_filter <";
            vertexset_where_expr->input_func->function_name->accept(this);
            oss << ">(";
            oss << vertexset_where_expr->target << ", ";
            vertexset_where_expr->input_func->accept(this);
            oss << ")";
        }
    }

    void CodeGenCPP::genEdgeSets() {
        for (auto edgeset : mir_context_->getEdgeSets()) {

            auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
            edge_set_type->accept(this);
            oss << edgeset->name << ";" << std::endl;

            // Deprecated code
//            if (edge_set_type->weight_type != nullptr) {
//                //weighted edgeset
//                //unweighted edgeset
//                oss << "WGraph " << edgeset->name << ";" << std::endl;
//            } else {
//                //unweighted edgeset
//                oss << "Graph " << edgeset->name << "; " << std::endl;
//            }
        }
    }


    /**
     * Generate the struct types before the arrays are generated
     */
    void CodeGenCPP::genStructTypeDecls() {
        for (auto const &struct_type_decl_entry : mir_context_->struct_type_decls) {
            auto struct_type_decl = struct_type_decl_entry.second;
            oss << "typedef struct ";
            oss << struct_type_decl->name << " { " << std::endl;

            for (auto var_decl : struct_type_decl->fields) {
                indent();
                printIndent();
                var_decl->type->accept(this);
                //we don't initialize in the struct declarations anymore
                // the initializations are done in the main function
                oss << var_decl->name;
                // << " = ";
                //var_decl->initVal->accept(this);
                oss << ";" << std::endl;
                dedent();
            }
            oss << "} " << struct_type_decl->name << ";" << std::endl;
        }
    }

    void CodeGenCPP::visit(mir::VertexSetType::Ptr vertexset_type) {
	if (vertexset_type->priority_update_type == mir::PriorityUpdateType::ExternPriorityUpdate || vertexset_type->priority_update_type == mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate)
	    oss << "julienne::vertexSubset ";
	else 
	    oss << "VertexSubset<int> *  ";
    }

    void CodeGenCPP::visit(mir::ListType::Ptr list_type) {
        oss << "std::vector< ";
        list_type->element_type->accept(this);
        oss << " > *  ";
    }

    void CodeGenCPP::visit(mir::NegExpr::Ptr neg_expr) {
        if (neg_expr->negate) oss << " -";
        neg_expr->operand->accept(this);
    }

    void CodeGenCPP::genEdgesetApplyFunctionCall(mir::EdgeSetApplyExpr::Ptr apply) {
        // the arguments order here has to be consistent with genEdgeApplyFunctionSignature in gen_edge_apply_func_decl.cpp

	
        auto edgeset_apply_func_name = edgeset_apply_func_gen_->genFunctionName(apply);
        oss << edgeset_apply_func_name << "(";

        apply->target->accept(this);
        oss << ", ";

        if (apply->from_func) {
            if (mir_context_->isFunction(apply->from_func->function_name->name)) {
                apply->from_func->accept(this);
            } else {
                // the input is an input from vertexset

                // TODO is it correct that we just accept the Identifier
                apply->from_func->function_name->accept(this);
            }

            oss << ", ";
        }

        if (apply->to_func) {
            if (mir_context_->isFunction(apply->to_func->function_name->name)) {
                apply->to_func->accept(this);
            } else {
                // the input is an input from vertexset

                // TODO is it correct that we just accept the Identifier
                apply->to_func->function_name->accept(this);
            }
            oss << ", ";
        }



        // the original apply function (pull direction in hybrid case)
        apply->input_function->accept(this);



        // a filter function for the push direction in hybrid code
        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)) {
            auto apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply);

            auto pushToFunctionExist = false;

            if (apply_expr->push_to_function_){
                oss << ", ";
                apply_expr->push_to_function_->accept(this);
                //TODO look at it again?
                //arguments.push_back(genFunctorNameAsArgumentString(apply_expr->push_to_function_, apply_expr->toFuncFunctorArgs));
                oss << ", ";
                pushToFunctionExist = true;

            }

            std::vector<std::string> pushFunctionArguments;
            //pushFunctionArguments.push_back(apply_expr->tracking_field);
            //TODO look at it again?
            //apply_expr->push_function_->functorArgs = apply_expr->input_function->functorArgs;
            if (pushToFunctionExist) {
                apply_expr->push_function_->accept(this);
            } else {
                oss << ", ";
                apply_expr->push_function_->accept(this);
            }

        }


        oss << "); " << std::endl;
    }


    void CodeGenCPP::visit(mir::IntersectionExpr::Ptr intersection_exp){

        if (intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::HIROSHI) {
            oss << "hiroshiVertexIntersection(";
        }

        else if(intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::MULTISKIP) {
            oss << "multiSkipVertexIntersection(";
        }

        else if(intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::COMBINED) {
            oss << "combinedVertexIntersection(";
        }

        else if(intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::BINARY) {
            oss << "binarySearchIntersection(";
        }

        else {
            oss << "naiveVertexIntersection(";
        }

        intersection_exp->vertex_a->accept(this);
        oss << ", ";
        intersection_exp->vertex_b->accept(this);
        oss << ", ";
        intersection_exp->numA->accept(this);
        oss << ", ";
        intersection_exp->numB->accept(this);
        // reference is an optional parameter only used for Triangular Counting
        if (intersection_exp->reference != nullptr){
            oss << ", ";
            intersection_exp->reference->accept(this);
        }
        oss << ") ";

    }

    void CodeGenCPP::visit(mir::IntersectNeighborExpr::Ptr intersection_exp){

        if (intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::HIROSHI) {
            oss << "hiroshiVertexIntersectionNeighbor(";
        }

        else if(intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::MULTISKIP) {
            oss << "multiSkipVertexIntersectionNeighbor(";
        }

        else if(intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::COMBINED) {
            oss << "combinedVertexIntersectionNeighbor(";
        }

        else if(intersection_exp->intersectionType == IntersectionSchedule::IntersectionType::BINARY) {
            oss << "binarySearchIntersectionNeighbor(";
        }

        else {
            oss << "naiveVertexIntersectionNeighbor(";
        }
        intersection_exp->edges->accept(this);
        oss << ", ";
        intersection_exp->vertex_a->accept(this);
        oss << ", ";
        intersection_exp->vertex_b->accept(this);
        oss << ") ";

    }

    void CodeGenCPP::visit(mir::EdgeSetLoadExpr::Ptr edgeset_load_expr) {
        if (edgeset_load_expr->is_weighted_) {
            oss << "builtin_loadWeightedEdgesFromFile ( ";
            edgeset_load_expr->file_name->accept(this);
            oss << ") ";
        } else {
            oss << "builtin_loadEdgesFromFile ( ";
            edgeset_load_expr->file_name->accept(this);
            oss << ") ";
        }
    }

    void CodeGenCPP::visit(mir::EdgeSetType::Ptr edgeset_type) {
        if (edgeset_type->weight_type != nullptr) {
            //weighted edgeset
            //unweighted edgeset
            oss << "WGraph ";
        } else {
            //unweighted edgeset
            oss << "Graph ";
        }
    }

    void CodeGenCPP::visit(mir::VectorAllocExpr::Ptr alloc_expr) {
        oss << "new ";

        if (alloc_expr->scalar_type != nullptr){
            alloc_expr->scalar_type->accept(this);
        } else if (alloc_expr->vector_type != nullptr){
            oss << alloc_expr->vector_type->toString();
        }
        oss << "[ ";
        //This is the current number of elements, but we need the range
        //alloc_expr->size_expr->accept(this);
        const auto size_expr = mir_context_->getElementCount(alloc_expr->element_type);
        if (size_expr != nullptr)
            size_expr->accept(this);
	else {
	    // This means it is a vector of constant size. The size_expr now directly holds the constant literal.
	    alloc_expr->size_expr->accept(this);
	}
        oss << "]";
    }


    bool CodeGenCPP::isLiteral(mir::Expr::Ptr expression) {

        bool isIntLiteral = mir::isa<mir::IntLiteral>(expression);
        bool isBoolLiteral = mir::isa<mir::BoolLiteral>(expression);
        bool isFloatLiteral = mir::isa<mir::FloatLiteral>(expression);
        bool isStringLiteral = mir::isa<mir::StringLiteral>(expression);

        bool negativeIntOrFloatLiteral = false;
        if (mir::isa<mir::NegExpr>(expression)) {
            auto negExpr = mir::to<mir::NegExpr>(expression);
            negativeIntOrFloatLiteral = mir::isa<mir::IntLiteral>(negExpr->operand) || mir::isa<mir::FloatLiteral>(negExpr->operand);
        }

        return isIntLiteral || isBoolLiteral || isFloatLiteral || isStringLiteral || negativeIntOrFloatLiteral;

    }



    void CodeGenCPP::genTypesRequiringTypeDefs() {

        for (mir::Type::Ptr type : mir_context_->types_requiring_typedef){
            if(mir::isa<mir::VectorType>(type)){
                auto vector_type = mir::to<mir::VectorType>(type);
                int range = vector_type->range_indexset;
                std::string typedef_name = vector_type->toString();
                if (mir_context_->defined_types.find(typedef_name) == mir_context_->defined_types.end()){
                    mir_context_->defined_types.insert(typedef_name);
                    //first generates a typedef for the vector type
                    oss << "typedef ";
                    vector_type->vector_element_type->accept(this);
                    oss << typedef_name <<  " ";
                    oss << "[ " << range << "]; " << std::endl;
                }
            }
        }
    }

    void CodeGenCPP::visit(mir::PriorityQueueType::Ptr priority_queue_type) {
        if (priority_queue_type->priority_update_type == mir::PriorityUpdateType::EagerPriorityUpdate
            || priority_queue_type->priority_update_type == mir::PriorityUpdateType::EagerPriorityUpdateWithMerge) {

            oss << "EagerPriorityQueue < ";
            priority_queue_type->priority_type->accept(this);
            oss << " >* ";

        } else if (priority_queue_type->priority_update_type == mir::PriorityUpdateType::ExternPriorityUpdate
        || priority_queue_type->priority_update_type == mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate
        || priority_queue_type->priority_update_type == mir::PriorityUpdateType::ReduceBeforePriorityUpdate) { // Add rest of the cases here as required
            oss << "julienne::PriorityQueue < ";
	    priority_queue_type->priority_type->accept(this);
	    oss << " >* ";
	} else {
           std::cout << "PriorityQueue type not supported yet" << std::endl;
        }
    }

    void CodeGenCPP::visit(mir::PriorityQueueAllocExpr::Ptr priority_queue_alloc_expr) {

        if (priority_queue_alloc_expr->priority_update_type == mir::PriorityUpdateType::EagerPriorityUpdate
            ||
            priority_queue_alloc_expr->priority_update_type == mir::PriorityUpdateType::EagerPriorityUpdateWithMerge) {


            oss << "new EagerPriorityQueue <";
            priority_queue_alloc_expr->priority_type->accept(this);
            oss << "> ( ";
            oss << priority_queue_alloc_expr->vector_function;


            if (priority_queue_alloc_expr->delta < 0 ){
                oss << ", stoi(argv[" << -1*priority_queue_alloc_expr->delta << "])";
            } else {
                oss << ", " << priority_queue_alloc_expr->delta;
            }
            oss << "); ";

        } else if (priority_queue_alloc_expr->priority_update_type == mir::PriorityUpdateType::ExternPriorityUpdate
        || priority_queue_alloc_expr->priority_update_type == mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate
        || priority_queue_alloc_expr->priority_update_type == mir::PriorityUpdateType::ReduceBeforePriorityUpdate) {  // Add other types here

            oss << "new julienne::PriorityQueue <";
            priority_queue_alloc_expr->priority_type->accept(this);
            oss << " > ( ";

            oss << mir_context_->getEdgeSets()[0]->name;

            if (priority_queue_alloc_expr->priority_update_type == mir::PriorityUpdateType::ReduceBeforePriorityUpdate){
                oss << ".num_nodes(), ";
            } else {
                oss << ".julienne_graph.n, ";
            }

            oss << priority_queue_alloc_expr->vector_function;
                oss << ", ";

            oss << "(julienne::bucket_order)";
                oss << priority_queue_alloc_expr->bucket_ordering;
                oss << ", ";

            oss << "(julienne::priority_order)";
            oss << priority_queue_alloc_expr->priority_ordering;
            oss << ", ";


            if (mir_context_->num_open_buckets < 0){
                oss << " stoi(argv[" << -1*mir_context_->num_open_buckets << "]) ";
            } else {
                oss << mir_context_->num_open_buckets ;
            }


            if (mir_context_->delta_ < 0){
                oss << ", stoi(argv[" << -1*mir_context_->delta_ << "]) ";
            } else {
                if (mir_context_->delta_ != 1){
                    oss << ", " << mir_context_->delta_;
                }
            }




            oss << ")";

	} else {
            std::cout << "PriorityQueue constructor not supported yet" << std::endl;
        }

    }

    void CodeGenCPP::visit(mir::OrderedProcessingOperator::Ptr ordered_op) {
        printIndent();
        if (ordered_op->priority_udpate_type == mir::PriorityUpdateType::EagerPriorityUpdate){
            oss << "OrderedProcessingOperatorNoMerge(";
        } else if (ordered_op->priority_udpate_type == mir::PriorityUpdateType::EagerPriorityUpdateWithMerge){
            oss << "OrderedProcessingOperatorWithMerge(";
        } else {
            std::cout << "Error: Unsupported Schedule for OrderedProcessingOperator" << std::endl;
        }
        oss << ordered_op->priority_queue_name << ", ";
        ordered_op->graph_name->accept(this);
        oss << ", ";

        //lambda function for while condition
        oss << "[&]()->bool{return (";
        ordered_op->while_cond_expr->accept(this);
        oss << ");}, ";

        //the user defined edge update function, instantiated with a functor
        // augmented with local_bins argument,
        ordered_op->edge_update_func->accept(this);
        oss << ", ";

        // supply the merge threshold argument for EagerPriorityUpdateWithMerge schedule
        if (ordered_op->priority_udpate_type == mir::PriorityUpdateType::EagerPriorityUpdateWithMerge){

            if (ordered_op->bucket_merge_threshold < 0){
                oss << " stoi(argv[" << -1*ordered_op->bucket_merge_threshold << "]), ";
            } else {
                oss << ordered_op->bucket_merge_threshold << ", ";
            }
        }

        ordered_op->optional_source_node->accept(this);

        oss << ");" << std::endl;
    }

    void CodeGenCPP::visit(mir::PriorityUpdateOperatorMin::Ptr priority_update_op) {

        if (mir_context_->priority_update_type == mir::EagerPriorityUpdate
        || mir_context_->priority_update_type == mir::EagerPriorityUpdateWithMerge){
            oss << priority_update_op->name;


            if (priority_update_op->generic_type != nullptr) {
                oss << " < ";
                priority_update_op->generic_type->accept(this);
                oss << " > ";
            }

            oss << "()"; //this should be a functor
            oss << "(";

            auto priority_queue_name_expr = priority_update_op->args[0];
            priority_queue_name_expr->accept(this);
            oss << ", ";


            if(mir_context_->priority_update_type == mir::PriorityUpdateType::EagerPriorityUpdateWithMerge ||
               mir_context_->priority_update_type ==  mir::PriorityUpdateType::EagerPriorityUpdate){
                // if this is a priority update edge function for EagerPriorityUpdate with and without merge
                // Then we need to insert an extra argument local bins
                oss << "local_bins, ";
            }

            bool printDelimiter = false;

            //start from index 1, so printed the first argument of priority queue name earlier
            for (int i = 1; i < priority_update_op->args.size(); i++) {
                auto arg = priority_update_op->args[i];
                if (printDelimiter) {
                    oss << ", ";
                }
                arg->accept(this);
                printDelimiter = true;
            }

            oss << ") ";
        } else if (mir_context_->priority_update_type == mir::ReduceBeforePriorityUpdate) {
            priority_update_op->priority_queue->accept(this);
            if (priority_update_op->is_atomic){
                oss << "->updatePriorityMinAtomic(";
            } else {
                oss << "->updatePriorityMin(";
            }
            priority_update_op->destination_node_id->accept(this);
            oss << ", ";
            priority_update_op->old_val->accept(this);
            oss << ", ";
            priority_update_op->new_val->accept(this);
            oss << ")";
        } else {
            std::cout << "updatePriorityMin not supported with this schedule"<< std::endl;
        }


    }


    void CodeGenCPP::visit(mir::UpdatePriorityExternCall::Ptr extern_call) {
//        printIndent();
//	oss << "{" << std::endl;
//	indent();
        printIndent();
	oss << "julienne::vertexSubset ";
	oss << extern_call->output_set_name; 
	oss << " = ";
	oss << extern_call->apply_function_name;
	oss << "( ";
	extern_call->input_set->accept(this);
	oss << ");" << std::endl;
	
	printIndent();
	oss << "auto ";
	oss << extern_call->lambda_name;
	oss << " = ";

	oss << "[&] (size_t i) -> julienne::Maybe<std::tuple<julienne::uintE, ";
	//mir::to<mir::PriorityQueueType>(mir_context_->getPriorityQueueDecl()->type)->priority_type->accept(this);

	oss << "julienne::uintE";
	oss << ">> {" << std::endl;
	
	indent();

	printIndent();
	oss << "const julienne::uintE v = ";
	extern_call->input_set->accept(this);
	oss << ".vtx(i);" << std::endl;
	
	printIndent();
	oss << "const julienne::uintE bkt = ";
	oss << extern_call->priority_queue_name;

	if (mir_context_->nodes_init_in_buckets){
        oss << "->get_bucket_no_overflow_insertion(";
	} else {
        oss << "->get_bucket_with_overflow_insertion(";
    }

	oss << mir_context_->priority_queue_alloc_list_[0]->vector_function;
	oss << "[v]);" << std::endl;

	printIndent();
	oss << "return julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>>(std::make_tuple(v, bkt));" << std::endl;
	
	dedent();
	printIndent();
	oss << "};" << std::endl;
	
	
//	dedent();
//	printIndent();
//	oss << "}" << std::endl;
    }

    void CodeGenCPP::visit(mir::UpdatePriorityUpdateBucketsCall::Ptr update_call) {
        printIndent();

        if (mir_context_->priority_update_type == mir::ConstSumReduceBeforePriorityUpdate
        || mir_context_->priority_update_type == mir::PriorityUpdateType::ExternPriorityUpdate){
            oss << update_call->priority_queue_name;
            oss << "->update_buckets(";
            oss << update_call->lambda_name;
            oss << ", ";
            oss << update_call->modified_vertexsubset_name;
            oss << ".size());" << std::endl;
        } else if (mir_context_->priority_update_type == mir::ReduceBeforePriorityUpdate){
            oss << "updateBucketWithGraphItVertexSubset(";
            oss << update_call->lambda_name << ", ";
            oss << update_call->priority_queue_name << ", ";
            oss << update_call->nodes_init_in_bucket << ", ";
            if (update_call->delta > 0){
                oss << update_call->delta;
            } else {
                oss << "stoi(argv[" << -1*update_call->delta << "])";
            }
            oss << ");" << std::endl;
        } else {
            std::cout << "UpdatePriorityUpdateBucketsCall not supported." << std::endl;
        }

    }

    void CodeGenCPP::get_edge_count_lambda(mir::UpdatePriorityEdgeCountEdgeSetApplyExpr::Ptr call) {
        //oss <<  "place_holder_edge_count_lambda";
	oss << "[&] (const tuple<julienne::uintE, julienne::uintE> &__p) {" << std::endl;
	indent();
	//printIndent();

	mir::FuncDecl::Ptr udf = mir_context_->getFunction(call->input_function->function_name->name);
	//udf->accept(this);

		
        //for (auto stmt : *(udf->body->stmts)) {
	for (auto stmt = udf->body->stmts->begin(); stmt != (udf->body->stmts->end()-1); stmt++){ 
		(*stmt)->accept(this);
	}

	mir::Stmt::Ptr stmt = *(udf->body->stmts->end()-1);
	assert(mir::isa<mir::ExprStmt>(stmt));
	mir::Expr::Ptr expr = mir::to<mir::ExprStmt>(stmt)->expr;
	
	assert(mir::to<mir::Call>(expr));
	mir::Call::Ptr call_expr = mir::to<mir::Call>(expr);
	
	mir::Expr::Ptr arg1 = call_expr->args[0];
	mir::Expr::Ptr arg2 = call_expr->args[1];
	mir::Expr::Ptr arg3 = call_expr->args[2];

	
	//arg3->accept(this);
	
	//if (call_expr->args.size() > 3)
	//	call_expr->args[3]->accept(this);


	if (call_expr->args.size() > 3) {
		mir::Expr::Ptr arg4 = call_expr->args[3];
		printIndent();			
		oss << "if ( ";
		oss << call->priority_queue_name;
		oss << "->get_tracking_variable()[std::get<0>(__p)]";
		oss << " > ";
		arg4->accept(this);
		oss << ") {" << std::endl;
		indent();
		printIndent();
		oss << "auto __new_pri = std::max(";
		oss << call->priority_queue_name;
		oss << "->get_tracking_variable()[std::get<0>(__p)] + ";
		arg3->accept(this);
		oss << "  * std::get<1>(__p), (unsigned int)(";
		arg4->accept(this);
		oss << "));" << std::endl;
		printIndent();
		oss << call->priority_queue_name;
		oss << "->get_tracking_variable()[std::get<0>(__p)] = __new_pri;" << std::endl;
		//oss << "->get_tracking_variable()[std::get<0>(__p)] = __new_pri;" << std::endl;
		printIndent();
		oss << "return julienne::wrap(std::get<0>(__p), ";
		oss << call->priority_queue_name;
		if (mir_context_->nodes_init_in_buckets){
            oss << "->get_bucket_no_overflow_insertion(__new_pri));" << std::endl;
		} else {
            oss << "->get_bucket_with_overflow_insertion(__new_pri));" << std::endl;
        }
		dedent();
		printIndent();
		oss << "}" << std::endl;
		printIndent();
		oss << "return julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>>();" << std::endl;
	} else {
		printIndent();
		
		oss << "auto __new_pri = ";
		oss << call->priority_queue_name;
		oss << "->get_tracking_variable()[std::get<0>(__p)] + ";
		arg3->accept(this);
		oss << "  * std::get<1>(__p);" << std::endl;
		printIndent();
		oss << call->priority_queue_name;
		oss << "->get_tracking_variable()[std::get<0>(__p)] = __new_pri;" << std::endl;
		//oss << "->get_tracking_variable()[std::get<0>(__p)] = __new_pri;" << std::endl;
		printIndent();
		oss << "return julienne::wrap(std::get<0>(__p), ";
		oss << call->priority_queue_name;
        if (mir_context_->nodes_init_in_buckets){
            oss << "->get_bucket_no_overflow_insertion(__new_pri));" << std::endl;
        } else {
            oss << "->get_bucket_with_overflow_insertion(__new_pri));" << std::endl;
        }

	}
	

	oss << std::endl;
	dedent();
	printIndent();
	oss << "}";
	
    }

    void CodeGenCPP::visit(mir::UpdatePriorityEdgeCountEdgeSetApplyExpr::Ptr call) {
	//oss << "< UpdatePriorityEdgeCountEdgeSetApplyExpr > ";


	oss << "julienne::vertexSubsetData<julienne::uintE> ";
	oss << call->moved_object_name;
	oss << " = ";
	call->target->accept(this);
	oss << ".em->edgeMapCount<julienne::uintE> (";
	//TODO this is bit hacky (should ideally accept funcexpr entirely)
	call->from_func->function_name->accept(this);
	oss << ", ";
	get_edge_count_lambda(call);
	oss << ")";
    }

    void CodeGenCPP::visit(mir::PriorityUpdateOperatorSum::Ptr update_op) {
        update_op->priority_queue->accept(this);
        if (update_op->is_atomic){
            oss << "->updatePrioritySumAtomic(";
        } else {
            oss << "->updatePrioritySum(";
        }
        update_op->destination_node_id->accept(this);
        oss << ", ";
        update_op->delta->accept(this);
        oss << ", ";
        update_op->minimum_val->accept(this);
        oss << ")";
    }

    void CodeGenCPP::genScalarVectorAlloc(mir::VarDecl::Ptr constant, mir::VectorType::Ptr vector_type) {
        oss << constant->name << " = new ";
        vector_type->vector_element_type->accept(this);
        oss << " ();" << std::endl;
    }


}