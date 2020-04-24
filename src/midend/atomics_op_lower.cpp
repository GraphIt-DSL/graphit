//
// Created by Yunming Zhang on 7/18/17.
//

#include <graphit/midend/atomics_op_lower.h>

void graphit::AtomicsOpLower::ApplyExprVisitor::visit(graphit::mir::PullEdgeSetApplyExpr::Ptr apply_expr) {
    singleFunctionEdgeSetApplyExprAtomicsLower(apply_expr);
}

void graphit::AtomicsOpLower::ApplyExprVisitor::visit(graphit::mir::PushEdgeSetApplyExpr::Ptr apply_expr) {
    singleFunctionEdgeSetApplyExprAtomicsLower(apply_expr);
}

void graphit::AtomicsOpLower::ApplyExprVisitor::visit(graphit::mir::HybridDenseForwardEdgeSetApplyExpr::Ptr apply_expr) {
    singleFunctionEdgeSetApplyExprAtomicsLower(apply_expr);
}

void graphit::AtomicsOpLower::ApplyExprVisitor::visit(graphit::mir::UpdatePriorityEdgeSetApplyExpr::Ptr apply_expr) {
    singleFunctionEdgeSetApplyExprAtomicsLower(apply_expr);
}


void graphit::AtomicsOpLower::ApplyExprVisitor::visit(graphit::mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr) {
    if (apply_expr->is_parallel){
        ReduceStmtLower reduce_stmt_lower = ReduceStmtLower(mir_context_);
        auto pull_func_name = apply_expr->input_function->function_name->name;
        mir::FuncDecl::Ptr pull_func_decl = mir_context_->getFunction(pull_func_name);
        auto push_func_name = apply_expr->push_function_->function_name->name;
        mir::FuncDecl::Ptr push_func_decl = mir_context_->getFunction(push_func_name);

        pull_func_decl->accept(&reduce_stmt_lower);
        push_func_decl->accept(&reduce_stmt_lower);

        lowerCompareAndSwap(apply_expr->to_func, apply_expr->from_func, apply_expr->input_function, apply_expr);
        lowerCompareAndSwap(apply_expr->to_func, apply_expr->from_func, apply_expr->push_function_, apply_expr);

    }
}

void graphit::AtomicsOpLower::ApplyExprVisitor::singleFunctionEdgeSetApplyExprAtomicsLower(graphit::mir::EdgeSetApplyExpr::Ptr apply_expr){
    if (apply_expr->is_parallel){
        ReduceStmtLower reduce_stmt_lower = ReduceStmtLower(mir_context_);
        auto apply_func_decl_name = apply_expr->input_function->function_name->name;
        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_func_decl_name);
        apply_func_decl->accept(&reduce_stmt_lower);
        lowerCompareAndSwap(apply_expr->to_func, apply_expr->from_func, apply_expr->input_function, apply_expr);
    }
}

void graphit::AtomicsOpLower::lower() {
    auto apply_expr_visitor = ApplyExprVisitor(mir_context_);
    std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
    for (auto function : functions) {
        function->accept(&apply_expr_visitor);
    }
}

bool graphit::AtomicsOpLower::ApplyExprVisitor::lowerCompareAndSwap(mir::FuncExpr::Ptr to_func,
                                                  mir::FuncExpr::Ptr from_func,
                                                  mir::FuncExpr::Ptr apply_func,
                                                  mir::EdgeSetApplyExpr::Ptr apply_expr) {
    if (to_func != nullptr){
        //pattern 1:
        // condition 1: to function reads field[v] (v is dst) (the only stmt in to_func), field read is tensorArrayRead
        // condition 2: apply function has one assignment only
        // condition 3: assignment is on the same field and on dst, and this is a shared write
        // condition 4: the data must be of a supported type

        auto to_func_decl = mir_context_->getFunction(to_func->function_name->name);
        auto to_func_body = to_func_decl->body;
        std::string compare_filed;
        // this is the value that the CAS is going to compare to if the CAS is to be generated
        mir::Expr::Ptr to_func_compare_val = nullptr;
        if (to_func_body->stmts->size() == 1){
            // condition 1: to function reads field[v] (v is dst) (the only stmt in to_func)
            mir::Stmt::Ptr only_stmt = (*(to_func_body->stmts))[0];
            // we expect the to func to be of the format output = field[v] == val
            if (mir::isa<mir::AssignStmt>(only_stmt)){
                mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(only_stmt);
                if (!mir::isa<mir::VarExpr>(assign_stmt->lhs)) return false;
                if (mir::to<mir::VarExpr>(assign_stmt->lhs)->var.getName() != to_func_decl->result.getName()){
                    //if the assignment do not update the return bool val, then abort
                    return false;
                }
                mir::Expr::Ptr rhs = assign_stmt->expr;
                if (!mir::isa<mir::EqExpr>(rhs)){
                    //if the rhs of the assignment is not a EqExpr, then abort
                    return false;
                }
                mir::EqExpr::Ptr eq_expr = mir::to<mir::EqExpr>(rhs);
                if (! mir::isa<mir::TensorArrayReadExpr>(eq_expr->operands[0]))
                    // the lhs has to be a tensor array read (tensor struct ready would not do)
                    return false;

                mir::TensorArrayReadExpr::Ptr tensor_array_read_expr
                        = mir::to<mir::TensorArrayReadExpr>(eq_expr->operands[0]);
                std::string field_name = tensor_array_read_expr->getTargetNameStr();
                mir::Type::Ptr field_type = mir_context_->getVectorItemType(field_name);
                if (mir::isa<mir::ScalarType>(field_type)){
                    mir::ScalarType::Ptr scalar_type = mir::to<mir::ScalarType>(field_type);
                    if (scalar_type->type == mir::ScalarType::Type::INT
                        || scalar_type->type == mir::ScalarType::Type::FLOAT){
                        // the tensor has to be of CAS compaitlbe type, currently we only do int and floats
                        // now we can set the expression
                        to_func_compare_val = eq_expr->operands[1];
                        compare_filed = field_name;
                    } else {
                        // not int or float
                        return false;
                    }
                } else {
                    // not a scalar type
                    return false;
                }
            }
        }

        if (to_func_compare_val == nullptr)
            // to_func did not fit the pattern
            return false;

        auto apply_func_decl = mir_context_->getFunction(apply_func->function_name->name);
        auto apply_func_body = apply_func_decl->body;

        if (apply_func_body->stmts->size() == 1){
            //only one statement
            mir::Stmt::Ptr only_stmt = (*(apply_func_body->stmts))[0];
            // we expect the to func to be of the format field[dst] = assign_val
            if (mir::isa<mir::AssignStmt>(only_stmt)) {
                mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(only_stmt);
                if (! mir::isa<mir::TensorArrayReadExpr>(assign_stmt->lhs)){
                    return false;
                }
                mir::TensorArrayReadExpr::Ptr tensor_array_read_expr =
                        mir::to<mir::TensorArrayReadExpr>(assign_stmt->lhs);

                std::string field_name = tensor_array_read_expr->getTargetNameStr();
                //TODO: here we assume the destination is named "dst", later we might need to update this logic
                std::string index = tensor_array_read_expr->getIndexNameStr();
                FieldVectorProperty field_vector_prop = tensor_array_read_expr->field_vector_prop_;
                mir::Type::Ptr field_type = mir_context_->getVectorItemType(field_name);

                //condition 3
                if (field_name == compare_filed
                    && index == "dst"
                    && field_vector_prop.access_type_ == FieldVectorProperty::AccessType::SHARED){
                    // condition 4 checks
                    if (mir::isa<mir::ScalarType>(field_type)){
                        mir::ScalarType::Ptr scalar_type = mir::to<mir::ScalarType>(field_type);
                        if (scalar_type->type == mir::ScalarType::Type::INT
                            || scalar_type->type == mir::ScalarType::Type::FLOAT){
                            // the tensor has to be of CAS compaitlbe type, currently we only do int and floats

                            // at this point, we should be ready to generate the CAS
                            mir::CompareAndSwapStmt::Ptr cas_stmt = std::make_shared<mir::CompareAndSwapStmt>();
                            cas_stmt->lhs = assign_stmt->lhs;
                            cas_stmt->expr = assign_stmt->expr;
                            cas_stmt->compare_val_expr = to_func_compare_val;

                            //replace the original assignment with the CAS
                            (*(apply_func_body->stmts)).pop_back();
                            (*(apply_func_body->stmts)).push_back(cas_stmt);

                            //remove the to function given that now we use the CAS

                            if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply_expr)){
                                //if this is hybrid, just remove the push_to, and keep the other to for pull
                                //TODO: this is a bit hacky, think about how to do it better later
                                //(mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply_expr))->push_to_function_ = "";
                                // maybe there is better way to do this
                                //TODO: ask Yunming about this?
                                (mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply_expr))->push_to_function_ = nullptr;

                            } else {
                                //apply_expr->to_func = "";
                                apply_expr->to_func = nullptr;
                            }
                            return true;


                        } else {
                            // not int or float
                            return false;
                        }
                    } else {
                        // not a scalar type
                        return false;
                    }
                }
            }
        }


    }

    if (from_func != nullptr){
        //TODO: support this other pattern
        //pattern 2:
        // condition 1: from function reads field[v] (v is src) (the only stmt in from_func)
        // condition 2: apply functioin has one assignment only
        // condition 3: assignment is on the same field and on src, and this is a shared write
        // condition 4: the data must be of a supported type

    }

    return false;

}



void graphit::AtomicsOpLower::ReduceStmtLower::visit(graphit::mir::ReduceStmt::Ptr reduce_stmt) {
   //if the lhs is a tensor array ready (tensor struct ready would not do)
    if (! mir::isa<mir::TensorArrayReadExpr>(reduce_stmt->lhs) ){
        return;
    }

    mir::TensorArrayReadExpr::Ptr tensor_array_read_expr =
            mir::to<mir::TensorArrayReadExpr>(reduce_stmt->lhs);

    //if it is not a scalar tensor read, then we can't lower to atomic operations
    if (! mir::isa<mir::VarExpr>(tensor_array_read_expr->target)){
        return;
    }

    std::string field_name = tensor_array_read_expr->getTargetNameStr();
    FieldVectorProperty field_vector_prop = tensor_array_read_expr->field_vector_prop_;
    mir::Type::Ptr field_type = mir_context_->getVectorItemType(field_name);

    //if the property is a shared (must be read_write since it is reduce stmt)
    if (field_vector_prop.access_type_ == FieldVectorProperty::AccessType::SHARED) {
        //check if it is an supported type for atomic operations
        if (mir::isa<mir::ScalarType>(field_type)){
            mir::ScalarType::Ptr scalar_type = mir::to<mir::ScalarType>(field_type);
            if (scalar_type->type == mir::ScalarType::Type::INT
                || scalar_type->type == mir::ScalarType::Type::FLOAT
                || scalar_type->type == mir::ScalarType::Type::DOUBLE) {
                //update the type to atomic op
                reduce_stmt->is_atomic_ = true;
                switch (reduce_stmt->reduce_op_){
                    case mir::ReduceStmt::ReductionOp::MIN:
                        reduce_stmt->reduce_op_ = mir::ReduceStmt::ReductionOp::ATOMIC_MIN;
                        break;
                    case mir::ReduceStmt::ReductionOp::SUM:
                        reduce_stmt->reduce_op_ = mir::ReduceStmt::ReductionOp::ATOMIC_SUM;
                        break;
                    default:
                        std::cout << "not supported for atomics" << std::endl;
                        exit(0);
                }
            }
        }
    }

}
