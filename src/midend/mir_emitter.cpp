//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/midend/mir_emitter.h>

namespace graphit {


    void MIREmitter::visit(fir::ConstDecl::Ptr const_decl) {

        addVarOrConst(const_decl, true);
    };

    void MIREmitter::visit(fir::VarDecl::Ptr var_decl) {
        addVarOrConst(var_decl, false);
        auto mir_var_decl = std::make_shared<mir::VarDecl>();
        mir_var_decl->name = var_decl->name->ident;
	if (var_decl->initVal != nullptr)
	        mir_var_decl->initVal = emitExpr(var_decl->initVal);
	else
		mir_var_decl->initVal = nullptr;
        mir_var_decl->type = emitType(var_decl->type);
        mir_var_decl->stmt_label = var_decl->stmt_label;

        retStmt = mir_var_decl;
    };

    void MIREmitter::visit(fir::ExprStmt::Ptr expr_stmt) {
        auto mir_expr_stmt = std::make_shared<mir::ExprStmt>();
        mir_expr_stmt->expr = emitExpr(expr_stmt->expr);
        mir_expr_stmt->stmt_label = expr_stmt->stmt_label;
        retStmt = mir_expr_stmt;
    }

    void MIREmitter::visit(fir::AssignStmt::Ptr assign_stmt) {
        auto mir_assign_stmt = std::make_shared<mir::AssignStmt>();
        //we only have one expression on the left hand size
        assert(assign_stmt->lhs.size() == 1);
        mir_assign_stmt->lhs = emitExpr(assign_stmt->lhs.front());
        mir_assign_stmt->expr = emitExpr(assign_stmt->expr);
        mir_assign_stmt->stmt_label = assign_stmt->stmt_label;
        retStmt = mir_assign_stmt;
    }


    void MIREmitter::visit(fir::ReduceStmt::Ptr reduce_stmt) {
        auto mir_reduce_stmt = std::make_shared<mir::ReduceStmt>();
        //we only have one expression on the left hand size
        assert(reduce_stmt->lhs.size() == 1);
        mir_reduce_stmt->lhs = emitExpr(reduce_stmt->lhs.front());
        mir_reduce_stmt->expr = emitExpr(reduce_stmt->expr);
        mir_reduce_stmt->stmt_label = reduce_stmt->stmt_label;
        switch (reduce_stmt->reduction_op) {
            case fir::ReduceStmt::ReductionOp::SUM:
                mir_reduce_stmt->reduce_op_ = mir::ReduceStmt::ReductionOp::SUM;
                break;

            case fir::ReduceStmt::ReductionOp::MIN:
                mir_reduce_stmt->reduce_op_ = mir::ReduceStmt::ReductionOp::MIN;
                break;

            //for now in the midend, MAX and ASYNC_MAX are the same.
            case fir::ReduceStmt::ReductionOp::MAX:
                mir_reduce_stmt->reduce_op_ = mir::ReduceStmt::ReductionOp::MAX;
                break;


            case fir::ReduceStmt::ReductionOp::ASYNC_MAX:
                mir_reduce_stmt->reduce_op_ = mir::ReduceStmt::ReductionOp::MAX;
                break;

            case fir::ReduceStmt::ReductionOp::ASYNC_MIN:
                mir_reduce_stmt->reduce_op_ = mir::ReduceStmt::ReductionOp::MIN;
                break;

        }
        retStmt = mir_reduce_stmt;
    }

    void MIREmitter::visit(fir::PrintStmt::Ptr print_stmt) {
        auto mir_print_stmt = std::make_shared<mir::PrintStmt>();
        //we support printing only one argument first
        assert(print_stmt->args.size() == 1);
        mir_print_stmt->expr = emitExpr(print_stmt->args.front());
        retStmt = mir_print_stmt;
    }

    void MIREmitter::visit(fir::BreakStmt::Ptr break_stmt) {
        auto mir_break_stmt = std::make_shared<mir::BreakStmt>();
        retStmt = mir_break_stmt;
    }

    void MIREmitter::visit(fir::SubExpr::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::SubExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::VertexSetType::Ptr vertex_set_type) {
        const auto mir_vertex_set_type = std::make_shared<mir::VertexSetType>();
        mir_vertex_set_type->element = std::dynamic_pointer_cast<mir::ElementType>(
                emitType(vertex_set_type->element));
        if (!mir_vertex_set_type) {
            std::cout << "Error in Emitting MIR VertexSetType " << std::endl;
            return;
        };
        retType = mir_vertex_set_type;
    }


    void MIREmitter::visit(fir::ListType::Ptr list_type) {
        const auto mir_list_set_type = std::make_shared<mir::ListType>();
        mir_list_set_type->element_type = std::dynamic_pointer_cast<mir::Type>(
                emitType(list_type->list_element_type));
        if (!mir_list_set_type) {
            std::cout << "Error in Emitting MIR VertexSetType " << std::endl;
            return;
        };
        retType = mir_list_set_type;
    }

    void MIREmitter::visit(fir::EdgeSetType::Ptr edge_set_type) {
        const auto mir_edgeset_type = std::make_shared<mir::EdgeSetType>();
        mir_edgeset_type->element = std::dynamic_pointer_cast<mir::ElementType>(
                emitType(edge_set_type->edge_element_type));
        if (!mir_edgeset_type) {
            std::cout << "Error in Emitting MIR EdgeSetType " << std::endl;
            return;
        };
        auto mir_vector_element_type_list = new std::vector<mir::ElementType::Ptr>();
        for (auto vertex_element_type : edge_set_type->vertex_element_type_list) {
            auto mir_vertex_element_type = emitType(vertex_element_type);
            mir_vector_element_type_list->push_back(
                    std::dynamic_pointer_cast<mir::ElementType>(mir_vertex_element_type));
        }
        mir_edgeset_type->vertex_element_type_list = mir_vector_element_type_list;
        // emit weight type if it is a weighted edgeset
        if (edge_set_type->weight_type != nullptr)
            mir_edgeset_type->weight_type = mir::to<mir::ScalarType>(emitType(edge_set_type->weight_type));
        retType = mir_edgeset_type;
    }

    void MIREmitter::visit(fir::ScalarType::Ptr type) {
        auto output = std::make_shared<mir::ScalarType>();

        switch (type->type) {
            case fir::ScalarType::Type::INT:
                output->type = mir::ScalarType::Type::INT;
                retType = output;
                break;
            case fir::ScalarType::Type::FLOAT:
                output->type = mir::ScalarType::Type::FLOAT;
                retType = output;
                break;
            case fir::ScalarType::Type::DOUBLE:
                output->type = mir::ScalarType::Type::DOUBLE;
                retType = output;
                break;
            case fir::ScalarType::Type::BOOL:
                output->type = mir::ScalarType::Type::BOOL;
                retType = output;
                break;
//            case ScalarType::Type::COMPLEX:
//                retType = ir::Complex;
//                break;
            case fir::ScalarType::Type::STRING:
                output->type = mir::ScalarType::Type::STRING;
                retType = output;
                break;
            default:
                std::cout << "visit(fir::ScalarType) unrecognized scalar type" << std::endl;
                unreachable;
                break;
        }
    }

    void MIREmitter::visit(fir::NDTensorType::Ptr ND_tensor_type) {
        const auto mir_vector_type = std::make_shared<mir::VectorType>();
        //TODO: fix the code to deal with emitting various different types
        if (ND_tensor_type->element != nullptr) {
            mir_vector_type->element_type = std::dynamic_pointer_cast<mir::ElementType>(
                    emitType(ND_tensor_type->element));
            assert(mir_vector_type->element_type != nullptr);
        }

        //element of a vector can be a non-scalar (vector type)
        mir_vector_type->vector_element_type = emitType(ND_tensor_type->blockType);
        if (ND_tensor_type->indexSets.size() == 1 &&
                fir::isa<fir::RangeIndexSet>(ND_tensor_type->indexSets[0])){
            auto range_index_set = fir::to<fir::RangeIndexSet>(ND_tensor_type->indexSets[0]);
            mir_vector_type->range_indexset = range_index_set->range;
        }
        assert(mir_vector_type->vector_element_type != nullptr);


        retType = mir_vector_type;
    }

    void MIREmitter::visit(fir::NameNode::Ptr name_node) {
        ctx->scope();
        auto mir_name_node = std::make_shared<mir::NameNode>();
        mir_name_node->stmt_label = name_node->stmt_label;
        mir_name_node->body = mir::to<mir::StmtBlock>(emitStmt(name_node->body));
        ctx->unscope();
        retStmt = mir_name_node;
    }

    void MIREmitter::visit(fir::ForStmt::Ptr for_stmt) {
        ctx->scope();
        auto mir_for_stmt = std::make_shared<mir::ForStmt>();
        mir_for_stmt->loopVar = for_stmt->loopVar->ident;
        auto loop_var_type = std::make_shared<mir::ScalarType>();
        loop_var_type->type = mir::ScalarType::Type::INT;
        auto mir_var = mir::Var(for_stmt->loopVar->ident, loop_var_type);
        //copy over the label from fir
        mir_for_stmt->stmt_label = for_stmt->stmt_label;
        ctx->addSymbol(mir_var);

        mir_for_stmt->body = mir::to<mir::StmtBlock>(emitStmt(for_stmt->body));
        mir_for_stmt->domain = emitDomain(for_stmt->domain);
        ctx->unscope();

        retStmt = mir_for_stmt;
    }

    void MIREmitter::visit(fir::RangeDomain::Ptr for_domain) {
        auto mir_for_domain = std::make_shared<mir::ForDomain>();
        mir_for_domain->upper = emitExpr(for_domain->upper);
        mir_for_domain->lower = emitExpr(for_domain->lower);
        retForDomain = mir_for_domain;
    }

    void MIREmitter::visit(fir::WhileStmt::Ptr while_stmt) {
        auto mir_while_stmt = std::make_shared<mir::WhileStmt>();
        mir_while_stmt->cond = emitExpr(while_stmt->cond);
        mir_while_stmt->body = mir::to<mir::StmtBlock>(emitStmt(while_stmt->body));
        mir_while_stmt->stmt_label = while_stmt->stmt_label;
        retStmt = mir_while_stmt;
    }

    void MIREmitter::visit(fir::IfStmt::Ptr if_stmt) {
        auto mir_if_stmt = std::make_shared<mir::IfStmt>();
        mir_if_stmt->cond = emitExpr(if_stmt->cond);
        mir_if_stmt->ifBody = emitStmt(if_stmt->ifBody);
        if (if_stmt->elseBody != nullptr)
            mir_if_stmt->elseBody = emitStmt(if_stmt->elseBody);
        retStmt = mir_if_stmt;
    }

    void MIREmitter::visit(fir::StmtBlock::Ptr stmt_block) {

        //initialize
        std::vector<mir::Stmt::Ptr> *stmts = new std::vector<mir::Stmt::Ptr>();

        for (auto stmt : stmt_block->stmts) {
            //stmt->accept(this);
            stmts->push_back(emitStmt(stmt));
        }

        auto output_stmt_block = std::make_shared<mir::StmtBlock>();
//        if (stmt_block->stmts.size() != 0) {
//            //output_stmt_block->stmts = ctx->getStatements();
//            output_stmt_block->stmts = stmts;
//        }
        output_stmt_block->stmts = stmts;

        retStmt = output_stmt_block;
    }

    void MIREmitter::visit(fir::VertexSetAllocExpr::Ptr expr) {
        // Currently we only record a size information in the MIR::VertexSetAllocExpr
        const auto mir_vertexsetalloc_expr = std::make_shared<mir::VertexSetAllocExpr>();
        mir_vertexsetalloc_expr->size_expr = emitExpr(expr->numElements);
        mir_vertexsetalloc_expr->element_type = mir::to<mir::ElementType>(emitType(expr->elementType));
        retExpr = mir_vertexsetalloc_expr;
    }


    void MIREmitter::visit(fir::ListAllocExpr::Ptr expr) {
        // Currently we only record a size information in the MIR::ListAllocExpr
        const auto mir_listalloc_expr = std::make_shared<mir::ListAllocExpr>();
        if (expr->numElements != nullptr)
            mir_listalloc_expr->size_expr = emitExpr(expr->numElements);
        mir_listalloc_expr->element_type = mir::to<mir::Type>(emitType(expr->general_element_type));
        retExpr = mir_listalloc_expr;
    }

    void MIREmitter::visit(fir::EdgeSetLoadExpr::Ptr load_expr) {
        auto mir_load_expr = std::make_shared<mir::EdgeSetLoadExpr>();
        mir_load_expr->file_name = emitExpr(load_expr->file_name);
        retExpr = mir_load_expr;
    }

    // use of variables
    void MIREmitter::visit(fir::VarExpr::Ptr var_expr) {
        auto mir_var_expr = std::make_shared<mir::VarExpr>();
        mir::Var associated_var = ctx->getSymbol(var_expr->ident);
        mir_var_expr->var = associated_var;
        retExpr = mir_var_expr;
    }

    void MIREmitter::visit(fir::AddExpr::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::AddExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::DivExpr::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::DivExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::NegExpr::Ptr neg_expr) {
        auto mir_expr = std::make_shared<mir::NegExpr>();
        mir_expr->negate = neg_expr->negate;
        mir_expr->operand = emitExpr(neg_expr->operand);
        retExpr = mir_expr;
    }

    void MIREmitter::visit(fir::EqExpr::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::EqExpr>();
        std::vector<mir::Expr::Ptr> mir_operands;
        for (auto expr : fir_expr->operands) {
            mir_operands.push_back(emitExpr(expr));
        }

        std::vector<mir::EqExpr::Op> mir_ops;
        for (unsigned i = 0; i < fir_expr->ops.size(); ++i) {
            mir::EqExpr::Op mir_op;

            switch (fir_expr->ops[i]) {
                case fir::EqExpr::Op::LT:
                    mir_op = mir::EqExpr::Op::LT;
                    break;
                case fir::EqExpr::Op::LE:
                    mir_op = mir::EqExpr::Op::LE;
                    break;
                case fir::EqExpr::Op::GT:
                    mir_op = mir::EqExpr::Op::GT;
                    break;
                case fir::EqExpr::Op::GE:
                    mir_op = mir::EqExpr::Op::GE;
                    break;
                case fir::EqExpr::Op::EQ:
                    mir_op = mir::EqExpr::Op::EQ;
                    break;
                case fir::EqExpr::Op::NE:
                    mir_op = mir::EqExpr::Op::NE;
                    break;
                default:
                    unreachable;
                    break;
            }
            mir_ops.push_back(mir_op);
        }


        mir_expr->ops = mir_ops;
        mir_expr->operands = mir_operands;
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::MulExpr::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::MulExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::CallExpr::Ptr call_expr) {
        auto mir_expr = std::make_shared<mir::Call>();
        mir_expr->name = call_expr->func->ident;
        std::vector<mir::Expr::Ptr> args;
        for (auto &fir_arg : call_expr->args) {
            const mir::Expr::Ptr mir_arg = emitExpr(fir_arg);
            args.push_back(mir_arg);
        }
        mir_expr->args = args;

        retExpr = mir_expr;
    };


    void MIREmitter::visit(fir::MethodCallExpr::Ptr method_call_expr) {
        auto mir_call_expr = std::make_shared<mir::Call>();

        if (std::dynamic_pointer_cast<fir::VarExpr>(method_call_expr->target) == nullptr) {
            std::cout << "error in emitting method call expression" << std::endl;
        }

        auto target_expr = std::dynamic_pointer_cast<fir::VarExpr>(method_call_expr->target);
        auto mir_target_var = ctx->getSymbol(target_expr->ident);
        auto mir_target_type = mir_target_var.getType();

        if (ctx->isConstVertexSet(target_expr->ident)) {
            // If target is a vertexset (vertexset is not an actual concrete object)
            if (method_call_expr->method_name->ident == "size") {
                // get the expression directly from the data structures if it is looking for size
                auto vertex_element_type = ctx->getElementTypeFromVectorOrSetName(target_expr->ident);
                retExpr = ctx->getElementCount(vertex_element_type);
            }

        } else if (mir::isa<mir::ListType>(mir_target_type)){
            //if this is a List type
            auto mir_list_type = mir::to<mir::ListType>(mir_target_type);
            mir_call_expr->generic_type = mir_list_type->element_type;
            mir_call_expr->name = method_call_expr->method_name->ident;

            std::vector<mir::Expr::Ptr> args;
            const auto self_arg = emitExpr(method_call_expr->target);
            //add the target to the argument
            args.push_back(self_arg);

            for (auto &fir_arg : method_call_expr->args) {
                const mir::Expr::Ptr mir_arg = emitExpr(fir_arg);
                args.push_back(mir_arg);
            }
            mir_call_expr->args = args;
            retExpr = mir_call_expr;

        } else {
            // If target is a vector or an edgeset (actual concrete object)

            mir_call_expr->generic_type = ctx->getVectorItemType(target_expr->ident);
            mir_call_expr->name = method_call_expr->method_name->ident;

            std::vector<mir::Expr::Ptr> args;
            const auto self_arg = emitExpr(method_call_expr->target);
            //add the target to the argument
            args.push_back(self_arg);

            //adds an additional argument for Builtin Sum
            if (method_call_expr->method_name->ident == "builtin_sum") {
                auto vertex_element_type = ctx->getElementTypeFromVectorOrSetName(target_expr->ident);
                const auto size_arg = ctx->getElementCount(vertex_element_type);
                args.push_back(size_arg);
            }

            for (auto &fir_arg : method_call_expr->args) {
                const mir::Expr::Ptr mir_arg = emitExpr(fir_arg);
                args.push_back(mir_arg);
            }
            mir_call_expr->args = args;
            retExpr = mir_call_expr;
        }
    }

    void MIREmitter::visit(fir::TensorReadExpr::Ptr tensor_read_expr) {
        auto mir_tensor_read_expr = std::make_shared<mir::TensorReadExpr>();
        mir_tensor_read_expr->target = emitExpr(tensor_read_expr->tensor);
        assert(tensor_read_expr->indices.size() == 1);
        if (!std::dynamic_pointer_cast<fir::ExprParam>(tensor_read_expr->indices.front())) {
            std::cout << "error in emitting mir TensorReadExpr, read param is not an ExprParam" << std::endl;
            return;
        }
        auto expr_param = std::dynamic_pointer_cast<fir::ExprParam>(tensor_read_expr->indices.front());
        mir_tensor_read_expr->index = emitExpr(expr_param->expr);
        retExpr = mir_tensor_read_expr;
    }


    void MIREmitter::visit(fir::ApplyExpr::Ptr apply_expr) {
        //dense vector apply
        auto target_expr = emitExpr(apply_expr->target);

        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(target_expr);
        if (!mir_var) {
            std::cout << "error in getting name of the vector in ApplyExpr" << std::endl;
            return;
        }

        //if (ctx->isConstVertexSet(mir_var->var.getName())) {
        if (mir::isa<mir::VertexSetType>(mir_var->var.getType())){
            //dense vertexset apply
            auto vertexset_apply_expr = std::make_shared<mir::VertexSetApplyExpr>();
            vertexset_apply_expr->target = target_expr;
            vertexset_apply_expr->input_function_name = apply_expr->input_function->ident;
            if (apply_expr->change_tracking_field != nullptr)
                vertexset_apply_expr->tracking_field = fir::to<fir::Identifier>(
                        apply_expr->change_tracking_field)->ident;
            retExpr = vertexset_apply_expr;
        }

        //if (ctx->isEdgeSet(mir_var->var.getName())) {
        //replace the isEdgeSet metadata, which was incomplete
        if(mir::isa<mir::EdgeSetType>(mir_var->var.getType())){

            auto edgeset_apply_expr = std::make_shared<mir::EdgeSetApplyExpr>();
            edgeset_apply_expr->target = target_expr;
            edgeset_apply_expr->input_function_name = apply_expr->input_function->ident;
            if (apply_expr->to_expr) edgeset_apply_expr->to_func = apply_expr->to_expr->input_func->ident;
            if (apply_expr->from_expr) {
                //TODO: move the checking from expr is a function or vertexsubset logic here
                edgeset_apply_expr->from_func = apply_expr->from_expr->input_func->ident;
            }
            if (apply_expr->change_tracking_field != nullptr)
                edgeset_apply_expr->tracking_field = fir::to<fir::Identifier>(apply_expr->change_tracking_field)->ident;
            if (apply_expr->disable_deduplication) edgeset_apply_expr->enable_deduplication = false;
            else edgeset_apply_expr->enable_deduplication = true;
            retExpr = edgeset_apply_expr;
        }

    }

    void MIREmitter::visit(fir::WhereExpr::Ptr where_expr) {
        //auto mir_where_expr = std::make_shared<mir::WhereExpr>();

        //target needs to be an varexpr
        //we use the varexpr to determine whether this is vertexset filtering or edgeset filtering

        auto fir_target_var_name = fir::to<fir::VarExpr>(where_expr->target)->ident;

        if (ctx->isConstVertexSet(fir_target_var_name)) {
            auto verteset_where_expr = std::make_shared<mir::VertexSetWhereExpr>();
            verteset_where_expr->target = fir_target_var_name;
            verteset_where_expr->input_func = where_expr->input_func->ident;
            verteset_where_expr->is_constant_set = true;
            retExpr = verteset_where_expr;
        }
    }

    void MIREmitter::visit(fir::ElementTypeDecl::Ptr element_type_decl) {
        const auto mir_element_type = std::make_shared<mir::ElementType>();
        mir_element_type->ident = element_type_decl->name->ident;
        addElementType(mir_element_type);
        retType = mir_element_type;
    }

    void MIREmitter::visit(fir::IdentDecl::Ptr ident_decl) {
        auto type = emitType(ident_decl->type);
        //TODO: add type info
        retVar = mir::Var(ident_decl->name->ident, type);
        //TODO: retField in the future ??
    }

    void MIREmitter::visit(fir::FuncDecl::Ptr func_decl) {
        auto mir_func_decl = std::make_shared<mir::FuncDecl>();
        ctx->scope();
        std::vector<mir::Var> arguments;

        //processing the arguments to the function declaration
        for (auto arg : func_decl->args) {
            const mir::Var arg_var = emitVar(arg);
            arguments.push_back(arg_var);
            ctx->addSymbol(arg_var);
        }
        mir_func_decl->args = arguments;

        //Processing the output of the function declaration
        //we assume there is only one argument for easy C++ code generation
        assert(func_decl->results.size() <= 1);
        if (func_decl->results.size()) {
            const mir::Var result_var = emitVar(func_decl->results.front());
            mir_func_decl->result = result_var;
            ctx->addSymbol(result_var);
        }

        //Processing the body of the function declaration
        mir::Stmt::Ptr body;
        if (func_decl->body != nullptr) {
            body = emitStmt(func_decl->body);

            //Aiming for not using the Scope Node
            //body = ir::Scope::make(body);
        }

        //cast the output to StmtBody
        mir_func_decl->body = std::dynamic_pointer_cast<mir::StmtBlock>(body);

        ctx->unscope();

        const auto func_name = func_decl->name->ident;
        mir_func_decl->name = func_name;
        //add the constructed function decl to functions
        ctx->addFunction(mir_func_decl);
    }

    void MIREmitter::visit(fir::BoolLiteral::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::BoolLiteral>();
        mir_expr->val = fir_expr->val;
        retExpr = mir_expr;
    };


    void MIREmitter::visit(fir::IntLiteral::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::IntLiteral>();
        mir_expr->val = fir_expr->val;
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::FloatLiteral::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::FloatLiteral>();
        mir_expr->val = fir_expr->val;
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::StringLiteral::Ptr fir_expr) {
        auto mir_expr = std::make_shared<mir::StringLiteral>();
        mir_expr->val = fir_expr->val;
        retExpr = mir_expr;
    };

    mir::Expr::Ptr MIREmitter::emitExpr(fir::Expr::Ptr ptr) {
        auto tmpExpr = retExpr;
        //we should get a null when we don't emit the right type of expr
        //retExpr = std::make_shared<mir::Expr>();

        ptr->accept(this);
        const mir::Expr::Ptr ret = retExpr;

        retExpr = tmpExpr;
        return ret;
    };

    mir::Stmt::Ptr MIREmitter::emitStmt(fir::Stmt::Ptr ptr) {
        auto tmpStmt = retStmt;
        retStmt = std::make_shared<mir::Stmt>();

        ptr->accept(this);
        const mir::Stmt::Ptr ret = retStmt;

        retStmt = tmpStmt;
        return ret;
    };

    mir::Type::Ptr MIREmitter::emitType(fir::Type::Ptr ptr) {
        auto tmpType = retType;

        ptr->accept(this);
        auto ret = retType;

        retType = tmpType;
        return ret;
    }


    mir::Var MIREmitter::emitVar(fir::IdentDecl::Ptr ptr) {
        auto tmpVar = retVar;

        ptr->accept(this);
        auto ret = retVar;

        retVar = tmpVar;
        return ret;
    }

    mir::ForDomain::Ptr MIREmitter::emitDomain(fir::ForDomain::Ptr ptr) {
        auto tmpDomain = retForDomain;
        retForDomain = std::make_shared<mir::ForDomain>();

        ptr->accept(this);
        const mir::ForDomain::Ptr ret = retForDomain;

        retForDomain = tmpDomain;
        return ret;
    }

    void MIREmitter::visit(fir::ElementType::Ptr element_type) {
        const auto mir_element_type = std::make_shared<mir::ElementType>();
        mir_element_type->ident = element_type->ident;
        retType = mir_element_type;
    }


    void MIREmitter::addVarOrConst(fir::VarDecl::Ptr var_decl, bool is_const) {
        //TODO: see if there is a cleaner way to do this, constructor may be???
        //construct a var decl variable
        const auto mir_var_decl = std::make_shared<mir::VarDecl>();
        if (var_decl->initVal)
            mir_var_decl->initVal = emitExpr(var_decl->initVal);
        mir_var_decl->name = var_decl->name->ident;
        mir_var_decl->type = emitType(var_decl->type);

        //construct a var variable
        const auto mir_var = mir::Var(mir_var_decl->name, mir_var_decl->type);
        ctx->addSymbol(mir_var);


        if (is_const) {

            if (std::dynamic_pointer_cast<mir::VectorType>(mir_var_decl->type) != nullptr) {
                mir::VectorType::Ptr type = std::dynamic_pointer_cast<mir::VectorType>(mir_var_decl->type);
                if (type->element_type != nullptr) {
                    // this is a field / system vector associated with an ElementType
                    ctx->updateVectorItemType(mir_var_decl->name, type->vector_element_type);
                    if (!ctx->updateElementProperties(type->element_type, mir_var_decl))
                        std::cout << "error in adding constant: " << mir_var_decl->name << std::endl;
                }
            } else if (std::dynamic_pointer_cast<mir::VertexSetType>(mir_var_decl->type) != nullptr) {
                mir::VertexSetType::Ptr type = std::dynamic_pointer_cast<mir::VertexSetType>(mir_var_decl->type);
                if (mir_var_decl->initVal != nullptr) {

                    if (mir::isa<mir::VertexSetAllocExpr>(mir_var_decl->initVal)) {
                        mir::VertexSetAllocExpr::Ptr vertexset_alloc_expr = mir::to<mir::VertexSetAllocExpr>(
                                mir_var_decl->initVal);
                        mir::Expr::Ptr size_expr = vertexset_alloc_expr->size_expr;
                        ctx->updateElementCount(type->element, size_expr);
                    } else {
                        ctx->updateElementCount(type->element, mir_var_decl->initVal);
                    }
                }
                //TODO: later may be fix this to vector or set name directly map to count
                ctx->setElementTypeWithVectorOrSetName(mir_var_decl->name, type->element);
                ctx->addConstVertexSet(mir_var_decl);
            } else if (std::dynamic_pointer_cast<mir::EdgeSetType>(mir_var_decl->type) != nullptr) {
                mir::EdgeSetType::Ptr type = std::dynamic_pointer_cast<mir::EdgeSetType>(mir_var_decl->type);
                if (mir_var_decl->initVal != nullptr) {
                    // the right hand side for edgeset const declartion needs to be a edgeset load expression
                    if (mir::isa<mir::EdgeSetLoadExpr>(mir_var_decl->initVal)) {
                        const auto init_val = mir::to<mir::EdgeSetLoadExpr>(mir_var_decl->initVal);
                        mir_var_decl->initVal = init_val;
                        ctx->updateElementInputFilename(type->element, init_val->file_name);
                        ctx->addEdgeSet(mir_var_decl);
                        ctx->addEdgesetType(mir_var_decl->name, type);
                    }
                }

            } else {
                mir_var_decl->modifier = "const";
                ctx->addConstant(mir_var_decl);
            }

        } else {
            //regular var decl
            //TODO:: need to figure out what we do here
        }
    }

    void MIREmitter::addElementType(mir::ElementType::Ptr element_type) {
        ctx->addElementType(element_type);
    }

}
