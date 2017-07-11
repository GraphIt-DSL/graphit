//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/codegen_cpp.h>

namespace graphit {
    int CodeGenCPP::genCPP() {
        genIncludeStmts();
        genEdgeSets();
        //genElementData();
        genStructTypeDecls();


        //Processing the constants
        for (auto constant : mir_context_->getLoweredConstants()) {
            if ((std::dynamic_pointer_cast<mir::VectorType>(constant->type)) != nullptr) {
                mir::VectorType::Ptr type = std::dynamic_pointer_cast<mir::VectorType>(constant->type);
                // if the constant decl is a field property of an element (system vector)
                if (type->element_type != nullptr) {
                    genPropertyArrayImplementationWithInitialization(constant);
                }
            } else if (std::dynamic_pointer_cast<mir::VertexSetType>(constant->type)) {
                // if the constant is a vertex set  decl
                // currently, no code is generated
            } else {
                // regular constant declaration
                constant->accept(this);
            }
        }

        //Generates function declarations for various edgeset apply operations with different schedules
        // TODO: actually complete the generation, fow now we will use libraries to test a few schedules
        // gen_edge_apply_function_visitor = EdgesetApplyFunctionDeclGenerator(mir_context_, oss);
        // gen_edge_apply_function_visitor.genEdgeApplyFuncDecls();

        //Processing the functions
        std::map<std::string, mir::FuncDecl::Ptr>::iterator it;
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

        for (auto it = functions.begin(); it != functions.end(); it++) {
            it->get()->accept(this);
        }


        oss << std::endl;
        return 0;
    };

    void CodeGenCPP::genIncludeStmts() {
        oss << "#include <iostream> " << std::endl;
        oss << "#include <vector>" << std::endl;
        oss << "#include \"intrinsics.h\"" << std::endl;
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
        printIndent();
        expr_stmt->expr->accept(this);
        if (!std::dynamic_pointer_cast<mir::ApplyExpr>(expr_stmt->expr)) {
            oss << ";";
        }
        oss << std::endl;
    }

    void CodeGenCPP::visit(mir::AssignStmt::Ptr assign_stmt) {

        if (mir::isa<mir::VertexSetWhereExpr>(assign_stmt->expr) ||
            mir::isa<mir::EdgeSetApplyExpr>(assign_stmt->expr)) {
            // declaring a new vertexset as output from where expression
            printIndent();
            assign_stmt->expr->accept(this);
            oss << std::endl;

            printIndent();

            assign_stmt->lhs->accept(this);
            oss << "  = ____graphit_tmp_out; " << std::endl;

        } else {
            printIndent();
            assign_stmt->lhs->accept(this);
            oss << " = ";
            assign_stmt->expr->accept(this);
            oss << ";" << std::endl;
        }
    }

    void CodeGenCPP::visit(mir::ReduceStmt::Ptr reduce_stmt) {

        if (mir::isa<mir::VertexSetWhereExpr>(reduce_stmt->expr) ||
            mir::isa<mir::EdgeSetApplyExpr>(reduce_stmt->expr)) {
            // declaring a new vertexset as output from where expression
            printIndent();
            reduce_stmt->expr->accept(this);
            oss << std::endl;

            printIndent();

            reduce_stmt->lhs->accept(this);
            oss << "  = ____graphit_tmp_out; " << std::endl;

        } else {
            switch (reduce_stmt->reduce_op_){
                case  mir::ReduceStmt::ReductionOp::SUM:
                    printIndent();
                    reduce_stmt->lhs->accept(this);
                    oss << " += ";
                    reduce_stmt->expr->accept(this);
                    oss << ";" << std::endl;
                    break;
                case  mir::ReduceStmt::ReductionOp::MIN:
                    printIndent();
                    oss << "if ( ( ";
                    reduce_stmt->lhs->accept(this);
                    oss << ") > ( " ;
                    reduce_stmt->expr->accept(this);
                    oss << ") ) { " << std::endl;
                    indent();
                    printIndent();
                    reduce_stmt->lhs->accept(this);
                    oss << "= ";
                    reduce_stmt->expr->accept(this);
                    oss << "; " << std::endl;


                    if (reduce_stmt->tracking_var_name_ != ""){
                        // need to generate a tracking variable
                        printIndent();
                        oss << reduce_stmt->tracking_var_name_ << " = true ; " << std::endl;
                    }

                    dedent();
                    printIndent();
                    oss << "} " << std::endl;
                    break;
                case  mir::ReduceStmt::ReductionOp::MAX:
                    oss << " max= ";
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

        if (mir::isa<mir::VertexSetWhereExpr>(var_decl->initVal) ||
            mir::isa<mir::EdgeSetApplyExpr>(var_decl->initVal)) {
            // declaring a new vertexset as output from where expression
            printIndent();
            var_decl->initVal->accept(this);
            oss << std::endl;

            printIndent();
            var_decl->type->accept(this);
            oss << var_decl->name << "  = ____graphit_tmp_out; " << std::endl;

        } else {
            printIndent();
            oss << var_decl->modifier << ' ';
            var_decl->type->accept(this);
            oss << var_decl->name << " ";
            if (var_decl->initVal != nullptr) {
                oss << "= ";
                var_decl->initVal->accept(this);
            }
            oss << ";" << std::endl;
        }
    }


    void CodeGenCPP::visit(mir::FuncDecl::Ptr func_decl) {

        //generate the return type
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

        } else if (func_decl->name == "main") {
            oss << "int ";
        } else {
            //default to void return type
            oss << "void ";
        }

        //generate the function name and left paren
        oss << func_decl->name << "(";

        bool printDelimiter = false;
        for (auto arg : func_decl->args) {
            if (printDelimiter) {
                oss << ", ";
            }

            arg.getType()->accept(this);
            oss << arg.getName();
            printDelimiter = true;
        }
        oss << ") ";



        //if the function has a body
        if (func_decl->body->stmts) {
            oss << std::endl;
            printBeginIndent();
            indent();

            func_decl->body->accept(this);

            //print a return statemetn if there is a result
            if (func_decl->result.isInitialized()) {
                printIndent();
                oss << "return " << func_decl->result.getName() << ";" << std::endl;
            }

            dedent();
            printEndIndent();
        }
        oss << ";";
        oss << std::endl;

    };

    void CodeGenCPP::visit(mir::ScalarType::Ptr scalar_type) {
        switch (scalar_type->type) {
            case mir::ScalarType::Type::INT:
                oss << "int ";
                break;
            case mir::ScalarType::Type::FLOAT:
                oss << "float ";
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

    void CodeGenCPP::visit(mir::StructTypeDecl::Ptr struct_type) {
        oss << struct_type->name << " ";
    }

    void CodeGenCPP::visit(mir::Call::Ptr call_expr) {
        oss << call_expr->name;


        if (call_expr->generic_type != nullptr) {
            oss << " < ";
            call_expr->generic_type->accept(this);
            oss << " > ";
        }
        oss << "(";

        bool printDelimiter = false;

        for (auto arg : call_expr->args) {
            if (printDelimiter) {
                oss << ", ";
            }
            arg->accept(this);
            printDelimiter = true;
        }

        oss << ") ";
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
        expr->target->accept(this);
        oss << "[";
        expr->index->accept(this);
        oss << "]";
    };

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

    void CodeGenCPP::visit(mir::VertexSetApplyExpr::Ptr apply_expr) {
        //vertexset apply
        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(apply_expr->target);
        auto associated_element_type = mir_context_->getElementTypeFromVectorOrSetName(mir_var->var.getName());
        assert(associated_element_type);
        auto associated_element_type_size = mir_context_->getElementCount(associated_element_type);
        assert(associated_element_type_size);
        oss << "for (int i = 0; i < ";
        associated_element_type_size->accept(this);
        oss << "; i++) {" << std::endl;
        indent();
        printIndent();
        oss << apply_expr->input_function_name << "(i);" << std::endl;
        dedent();
        printIndent();
        oss << "}";
    }

    void CodeGenCPP::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {

        // get the name of the function declaration
        auto func_name = edgeset_apply_func_gen_->genFunctionName(apply_expr);

        //edgeset apply
        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(apply_expr->target);
        //generate code for pull edgeset apply
        genEdgeSetPullApply(mir_var, apply_expr);
    }

    void CodeGenCPP::visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr) {
        // generate code for push edgeset apply
        genEdgesetPushApply(apply_expr);
    }

    void CodeGenCPP::visit(mir::VertexSetWhereExpr::Ptr vertexset_where_expr) {


        //dense vertex set apply
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
            vertices_range_expr->accept(this);
            oss << " );"  << std::endl;

            printIndent();
            oss << "for (int v = 0; v < ";
            associated_element_type_size->accept(this);
            oss << "; v++) {" << std::endl;
            indent();
            printIndent();
            oss << "if ( " << vertexset_where_expr->input_func << "( v ) )" << std::endl;
            indent();
            printIndent();
            oss << "____graphit_tmp_out->addVertex(v);" << std::endl;
            dedent();
            dedent();
            printIndent();
            oss << "}";
        }

    }

    void CodeGenCPP::genEdgeSets() {
        for (auto edgeset : mir_context_->getEdgeSets()) {

            auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset->type);
            if (edge_set_type->weight_type != nullptr){
                //weighted edgeset
                //unweighted edgeset
                oss << "WGraph " << edgeset->name << " = builtin_loadWeightedEdgesFromFile ( ";
                edgeset->initVal->accept(this);
                oss << " ); " << std::endl;

            } else {
                //unweighted edgeset
                oss << "Graph " << edgeset->name << " = builtin_loadEdgesFromFile ( ";
                edgeset->initVal->accept(this);
                oss << " ); " << std::endl;
            }


        }
    }

    void CodeGenCPP::genEdgeSetPullApply(mir::VarExpr::Ptr var_expr, mir::EdgeSetApplyExpr::Ptr apply_expr) {
        auto edgeset_name = var_expr->var.getName();
        bool apply_expr_gen_frontier = false;

        // Check if the apply function has a return value
        auto apply_func = mir_context_->getFunction(apply_expr->input_function_name);

        // For now, the temporary vertexset would be set at the number of assoicated vertices of Vertex type
        // Need to get the Vertex end point type from the edgeset, probably through the Edge type
        mir::VarExpr::Ptr edgeset_expr = mir::to<mir::VarExpr>(apply_expr->target);
        mir::VarDecl::Ptr edgeset_var_decl = mir_context_->getConstEdgeSetByName(edgeset_name);
        mir::EdgeSetType::Ptr edgeset_type = mir::to<mir::EdgeSetType>(edgeset_var_decl->type);
        assert(edgeset_type->vertex_element_type_list->size() == 2);
        mir::ElementType::Ptr dst_vertex_type = (*(edgeset_type->vertex_element_type_list))[1];
        auto dst_vertices_range_expr = mir_context_->getElementCount(dst_vertex_type);

        bool is_weighted_edgeset = false;
        if (edgeset_type->weight_type != nullptr){
            is_weighted_edgeset = true;
        }

        // If apply function has a return value, then we need to return a temporary vertexsubset
        if (apply_func->result.isInitialized()) {
            // build an empty vertex subset if apply function returns
            apply_expr_gen_frontier = true;
            oss << "auto ____graphit_tmp_out = new VertexSubset <NodeID> ( ";
            dst_vertices_range_expr->accept(this);
            oss << " , 0 );" << std::endl;

        } else {
            // If no return value is specified for the apply function, then it would return the entire set
            auto dst_vertices_range_expr = mir_context_->getElementCount(dst_vertex_type);
            oss << "auto ____graphit_tmp_out = new VertexSubset <NodeID> ( ";
            dst_vertices_range_expr->accept(this);
            oss << " , ";
            dst_vertices_range_expr->accept(this);
            oss << " );" << std::endl;
        }

        printIndent();
        oss << "for (NodeID d=0; d < " << edgeset_name << ".num_nodes(); d++) {" << std::endl;
        indent();
        printIndent();

        if (apply_expr->to_func != "") {
            oss << "if (" << apply_expr->to_func << "( d ) ) { " << std::endl;
            indent();
        }

        printIndent();

        if (! is_weighted_edgeset){
            oss << "for (NodeID s : " << edgeset_name << ".in_neigh(d)) {" << std::endl;
        } else {
            oss << "for (WNode s : " << edgeset_name << ".in_neigh(d)) {" << std::endl;
        }

        indent();
        printIndent();

        // print the checks on filtering on sources s
        if (apply_expr->from_func != "") {
            oss << "if ";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply_expr->from_func)) {
                //if the input expression is a function call
                oss << " ( " << apply_expr->from_func << " ( s )";

            } else {
                //the input expression is a vertex subset
                oss << " ( " << apply_expr->from_func << "->contains( s ) ";
            }


            oss << ") { " << std::endl;
        }

        indent();
        printIndent();
        if (apply_expr_gen_frontier) {
            oss << "if ( ";
        }

        // generating the C++ code for the apply function call
        if (is_weighted_edgeset){
            oss << apply_expr->input_function_name << "( s.v , d, s.w )";
        } else {
            oss << apply_expr->input_function_name << "( s , d  )";

        }

        if (!apply_expr_gen_frontier) {
            // no need to generate a frontier
            oss << ";" << std::endl;
        } else {
            //generate the code for adding destination to "next" frontier
            oss << " ) { ____graphit_tmp_out->addVertex(d); }" << std::endl;
        }

        // generating code for early break
        if (apply_expr->to_func != "") {
            printIndent();
            oss << "if (!" << apply_expr->to_func << "( d ) ) break; " << std::endl;
        }

        // end of from filtering
        if (apply_expr->from_func != "") {
            dedent();
            printIndent();
            oss << "}" << std::endl;
        }

        //end of for loop on the neighbors
        dedent();
        printIndent();
        oss << "}" << std::endl;

        if (apply_expr->to_func != "") {
            dedent();
            printIndent();
            oss << "} " << std::endl;
        }


        dedent();
        printIndent();
        oss << "}" << std::endl;
    }


    void CodeGenCPP::genEdgesetPushApply(mir::PushEdgeSetApplyExpr::Ptr apply_expr) {
        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(apply_expr->target);
        auto edgeset_name = mir_var->var.getName();
        bool apply_expr_gen_frontier = false;

        // Check if the apply function has a return value
        auto apply_func = mir_context_->getFunction(apply_expr->input_function_name);

        // For now, the temporary vertexset would be set at the number of assoicated vertices of Vertex type
        // Need to get the Vertex end point type from the edgeset, probably through the Edge type
        mir::VarExpr::Ptr edgeset_expr = mir::to<mir::VarExpr>(apply_expr->target);
        mir::VarDecl::Ptr edgeset_var_decl = mir_context_->getConstEdgeSetByName(edgeset_name);
        mir::EdgeSetType::Ptr edgeset_type = mir::to<mir::EdgeSetType>(edgeset_var_decl->type);
        assert(edgeset_type->vertex_element_type_list->size() == 2);
        mir::ElementType::Ptr dst_vertex_type = (*(edgeset_type->vertex_element_type_list))[1];
        auto dst_vertices_range_expr = mir_context_->getElementCount(dst_vertex_type);


        // If apply function has a return value, then we need to return a temporary vertexsubset
        if (apply_func->result.isInitialized()) {
            // build an empty vertex subset if apply function returns
            apply_expr_gen_frontier = true;
            oss << "auto ____graphit_tmp_out = new VertexSubset <NodeID> ( ";
            dst_vertices_range_expr->accept(this);
            oss << " );" << std::endl;

        } else {
            // If no return value is specified for the apply function, then it would return the entire set
            auto dst_vertices_range_expr = mir_context_->getElementCount(dst_vertex_type);
            oss << "auto ____graphit_tmp_out = new VertexSubset <NodeID> ( ";
            dst_vertices_range_expr->accept(this);
            oss << " , true);" << std::endl;
        }

        printIndent();
        oss << "for (NodeID s=0; s < " << edgeset_name << ".num_nodes(); s++) {" << std::endl;
        indent();
        printIndent();

        if (apply_expr->from_func != "") {
            oss << "if ";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply_expr->from_func)) {
                //if the input expression is a function call
                oss << " ( " << apply_expr->from_func << " ( s )";

            } else {
                //the input expression is a vertex subset
                oss << " ( " << apply_expr->from_func << "->contains( s ) ";
            }
            oss << ") { " << std::endl;
            indent();
        }

        printIndent();
        oss << "for (NodeID d : " << edgeset_name << ".out_neigh(s)) {" << std::endl;
        indent();
        printIndent();

        // print the checks on filtering on sources s
        if (apply_expr->to_func != "") {
            oss << "if ";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply_expr->to_func)) {
                //if the input expression is a function call
                oss << " ( " << apply_expr->to_func << " ( d )";

            } else {
                //the input expression is a vertex subset
                oss << " ( " << apply_expr->to_func << "->contains( d ) ";
            }


            oss << ") { " << std::endl;
        }

        indent();
        printIndent();
        if (apply_expr_gen_frontier) {
            oss << "if ( ";
        }

        // generating the C++ code for the apply function call

        oss << apply_expr->input_function_name << "( s , d )";

        if (!apply_expr_gen_frontier) {
            // no need to generate a frontier
            oss << ";" << std::endl;
        } else {
            //generate the code for adding destination to "next" frontier
            oss << " ) { ____graphit_tmp_out->addVertex(d); }" << std::endl;
        }

        // generating code for early break if source changed
        if (apply_expr->from_func != "") {
            printIndent();
            oss << "if ";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply_expr->from_func)) {
                //if the input expression is a function call
                oss << " ( " << apply_expr->from_func << " ( s )";

            } else {
                //the input expression is a vertex subset
                oss << " ( " << apply_expr->from_func << "->contains( s ) ";
            }
            oss << ") break; " << std::endl;
        }

        // end of from filtering
        if (apply_expr->from_func != "") {
            dedent();
            printIndent();
            oss << "}" << std::endl;
        }

        //end of for loop on the neighbors
        dedent();
        printIndent();
        oss << "}" << std::endl;

        if (apply_expr->to_func != "") {
            dedent();
            printIndent();
            oss << "} " << std::endl;
        }


        dedent();
        printIndent();
        oss << "}" << std::endl;
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
                oss << var_decl->name << " = ";
                var_decl->initVal->accept(this);
                oss << ";" << std::endl;
                dedent();
            }

            oss << "} " << struct_type_decl->name << ";" << std::endl;
        }
    }

    void CodeGenCPP::visit(mir::VertexSetType::Ptr vertexset_type) {
        oss << "VertexSubset<int> *  ";
    }

    void CodeGenCPP::visit(mir::NegExpr::Ptr neg_expr) {
        if (neg_expr->negate) oss << " -";
        neg_expr->operand->accept(this);
    }


}
