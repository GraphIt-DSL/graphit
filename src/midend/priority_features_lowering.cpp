//
// Created by Yunming Zhang on 3/20/19.
//

#include <graphit/midend/priority_features_lowering.h>

namespace graphit {

    void PriorityFeaturesLower::lower() {

        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

        // find the schedules specified for priority updates,
        // assuming we only have one priority queue at the moment
        auto schedule_finder = PriorityUpdateScheduleFinder(mir_context_, schedule_);
        //this visitor sets the priorty update type, and delta in mir_context
        for (auto function : functions) {
            function->accept(&schedule_finder);
        }


        // lowers the extern apply expression - This has to be done before lower_priority_queue_type_and_alloc_expr because this gathers some parameters into mir_context_ like the schedules
        auto lower_extern_apply_expr = LowerUpdatePriorityExternVertexSetApplyExpr(schedule_, mir_context_);
        for (auto function : functions) {
            lower_extern_apply_expr.rewrite(function);
        }


        // lowers the Priority Queue Type, and Alloc Expr based on the schedule
        auto lower_priority_queue_type_and_alloc_expr = LowerPriorityRelatedTypeAndExpr(mir_context_, schedule_);

        for (auto constant : mir_context_->getConstants()) {
            constant->accept(&lower_priority_queue_type_and_alloc_expr);
        }
        for (auto constant : mir_context_->const_edge_sets_) {
            constant->accept(&lower_priority_queue_type_and_alloc_expr);
        }
        for (auto function : functions) {
            function->accept(&lower_priority_queue_type_and_alloc_expr);
        }
        for (auto function : mir_context_->getExternFunctionList()) {
            function->accept(&lower_priority_queue_type_and_alloc_expr);
        }

        LowerUpdatePriorityEdgeSetApplyExpr lower_update_priority_edge_set_apply_expr = LowerUpdatePriorityEdgeSetApplyExpr(
                schedule_, mir_context_);

        for (auto function : functions) {
            function->accept(&lower_update_priority_edge_set_apply_expr);
        }

        // Detect pattern for OrderedProcessingOperator, and lower into the MIR node for OrderedProcessingOp
        auto lower_ordered_processing_op = LowerIntoOrderedProcessingOperatorRewriter(schedule_, mir_context_);
        for (auto function : functions) {
            lower_ordered_processing_op.rewrite(function);
        }

        // Lowers into PriorityUpdateOperators (PriorityUpdateMin and PriorityUpdateSum)
        auto lower_priority_update_rewriter = LowerPriorityUpdateOperatorRewriter(schedule_, mir_context_);
        for (auto function : functions) {
            lower_priority_update_rewriter.rewrite(function);
        }

        //lowers for the ReduceBeforeUpdate default schedule
        if (mir_context_->priority_update_type == mir::PriorityUpdateType::ReduceBeforePriorityUpdate) {
            auto lower_reduce_before_update = LowerReduceBeforePriorityUpdate(schedule_, mir_context_);
            for (auto function : functions) {
                function->accept(&lower_reduce_before_update);
            }
        }
    }

    void PriorityFeaturesLower::PriorityUpdateScheduleFinder::visit(
            mir::UpdatePriorityEdgeSetApplyExpr::Ptr update_priority_edgeset_apply_expr) {
        if (schedule_ != nullptr && schedule_->apply_schedules != nullptr) {
            auto current_label = label_scope_.getCurrentScope();
            setPrioritySchedule(current_label);
        }
    }

    void PriorityFeaturesLower::PriorityUpdateScheduleFinder::visit(
            mir::UpdatePriorityExternVertexSetApplyExpr::Ptr update_priority_extern_vertexset_apply_expr) {
        if (schedule_ != nullptr && schedule_->apply_schedules != nullptr) {
            auto current_label = label_scope_.getCurrentScope();
            setPrioritySchedule(current_label);
        }
    }

    void PriorityFeaturesLower::PriorityUpdateScheduleFinder::setPrioritySchedule(std::string current_label) {
        auto apply_schedule = schedule_->apply_schedules->find(current_label);
        if (apply_schedule != schedule_->apply_schedules->end()) { //a schedule is found

            if (apply_schedule->second.delta != 1) {
                mir_context_->delta_ = apply_schedule->second.delta;
            }


            if (apply_schedule->second.priority_update_type
                == ApplySchedule::PriorityUpdateType::REDUCTION_BEFORE_UPDATE) {
                mir_context_->priority_update_type = mir::PriorityUpdateType::ReduceBeforePriorityUpdate;
                mir_context_->num_open_buckets = apply_schedule->second.num_open_buckets;
            } else if (apply_schedule->second.priority_update_type
                       == ApplySchedule::PriorityUpdateType::EAGER_PRIORITY_UPDATE) {
                mir_context_->priority_update_type = mir::PriorityUpdateType::EagerPriorityUpdate;
            } else if (apply_schedule->second.priority_update_type
                       == ApplySchedule::PriorityUpdateType::CONST_SUM_REDUCTION_BEFORE_UPDATE) {
                mir_context_->priority_update_type = mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate;
                mir_context_->num_open_buckets = apply_schedule->second.num_open_buckets;
            } else if (apply_schedule->second.priority_update_type
                       == ApplySchedule::PriorityUpdateType::EAGER_PRIORITY_UPDATE_WITH_MERGE) {
                mir_context_->priority_update_type = mir::PriorityUpdateType::EagerPriorityUpdateWithMerge;

                if (apply_schedule->second.merge_threshold != 0) {
                    mir_context_->bucket_merge_threshold_ = apply_schedule->second.merge_threshold;
                }
            } else {
                mir_context_->priority_update_type = mir::PriorityUpdateType::NoPriorityUpdate;
            }
        } else {
            mir_context_->priority_update_type = mir::PriorityUpdateType::NoPriorityUpdate;
        }

    }

    void PriorityFeaturesLower::LowerUpdatePriorityExternVertexSetApplyExpr::visit(mir::ExprStmt::Ptr expr_stmt) {
        if (mir::isa<mir::UpdatePriorityExternVertexSetApplyExpr>(expr_stmt->expr)) {

            mir::to<mir::PriorityQueueType>(
                    mir_context_->getPriorityQueueDecl()->type)->priority_update_type = mir::PriorityUpdateType::ExternPriorityUpdate;
            mir_context_->priority_update_type = mir::PriorityUpdateType::ExternPriorityUpdate;

            mir::UpdatePriorityExternVertexSetApplyExpr::Ptr expr = mir::to<mir::UpdatePriorityExternVertexSetApplyExpr>(
                    expr_stmt->expr);

            mir::UpdatePriorityExternCall::Ptr call_stmt = std::make_shared<mir::UpdatePriorityExternCall>();
            call_stmt->input_set = expr->target;
            call_stmt->apply_function_name = expr->input_function->function_name->name;
            call_stmt->lambda_name = "generate_lambda_function_" + mir_context_->getUniqueNameCounterString();
            call_stmt->output_set_name = "generated_vertex_subset_" + mir_context_->getUniqueNameCounterString();
            call_stmt->priority_queue_name = mir_context_->getPriorityQueueDecl()->name;


            mir::UpdatePriorityUpdateBucketsCall::Ptr update_call = std::make_shared<mir::UpdatePriorityUpdateBucketsCall>();
            update_call->lambda_name = call_stmt->lambda_name;
            update_call->modified_vertexsubset_name = call_stmt->output_set_name;
            update_call->priority_queue_name = call_stmt->priority_queue_name;

            mir::StmtBlock::Ptr stmt_block = std::make_shared<mir::StmtBlock>();
            stmt_block->insertStmtEnd(call_stmt);
            stmt_block->insertStmtEnd(update_call);

            node = stmt_block;
            return;
        }
        node = expr_stmt;
    }

    void PriorityFeaturesLower::LowerIntoOrderedProcessingOperatorRewriter::visit(mir::WhileStmt::Ptr while_stmt) {
        // check if it matches the pattern
        if (checkWhileStmtPattern(while_stmt)) {
            // if matches the pattern, then replace with a separate operator
            mir::OrderedProcessingOperator::Ptr ordered_op = std::make_shared<mir::OrderedProcessingOperator>();
            ordered_op->while_cond_expr = while_stmt->cond;
            //get the UpdatePriorityEdgesetApply label, for retrieving the schedule
            auto stmt_blk = while_stmt->body;
            mir::Stmt::Ptr first_stmt = (*(stmt_blk->stmts))[0];
            mir::Stmt::Ptr second_stmt = (*(stmt_blk->stmts))[1];
            mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>(second_stmt);
            auto update_priority_edgesetapply_expr = mir::to<mir::UpdatePriorityEdgeSetApplyExpr>(expr_stmt->expr);

            auto edge_update_func_name = update_priority_edgesetapply_expr->input_function->function_name->name;
            auto identDecl = std::make_shared<mir::IdentDecl>();
            identDecl->name = edge_update_func_name;
            auto funcExpr = std::make_shared<mir::FuncExpr>();
            funcExpr->function_name = identDecl;
            funcExpr->functorArgs = update_priority_edgesetapply_expr->input_function->functorArgs;
            ordered_op->edge_update_func = funcExpr;

            auto graph_name = update_priority_edgesetapply_expr->target;
            mir_context_->eager_priority_update_edge_function_name = edge_update_func_name;

            ordered_op->graph_name = graph_name;

            //auto priority_queue_name;
            std::string priority_queue_name = mir_context_->getPriorityQueueDecl()->name;
            ordered_op->priority_queue_name = priority_queue_name;
            ordered_op->optional_source_node = mir_context_->optional_starting_source_node;

            if (mir_context_->priority_update_type == mir::PriorityUpdateType::EagerPriorityUpdateWithMerge) {
                ordered_op->priority_udpate_type = mir::PriorityUpdateType::EagerPriorityUpdateWithMerge;
                ordered_op->bucket_merge_threshold = mir_context_->bucket_merge_threshold_;
            } else {
                ordered_op->priority_udpate_type = mir::PriorityUpdateType::EagerPriorityUpdate;
            }


            //use the schedule to set
            node = ordered_op;
        } else {
            node = while_stmt;
        }
    }


    // One example of a pattern that we are detecting
    //"  while (pq.finished() == false) "
    //"    var frontier : vertexsubset = pq.dequeue_ready_set(); % dequeue_ready_set() \n"
    //"    #s1# edges.from(frontier).applyUpdatePriority(updateEdge);  \n"
    //"    delete frontier; "

    bool PriorityFeaturesLower::LowerIntoOrderedProcessingOperatorRewriter::checkWhileStmtPattern(
            mir::WhileStmt::Ptr while_stmt) {
        auto stmt_blk = while_stmt->body;
        int num_stmts = (*(stmt_blk->stmts)).size();

        // for now assuming
        if (num_stmts < 2 || num_stmts > 3) {
            return false;
        }

        mir::Stmt::Ptr first_stmt = (*(stmt_blk->stmts))[0];
        mir::Stmt::Ptr second_stmt = (*(stmt_blk->stmts))[1];


        if (mir::isa<mir::VarDecl>(first_stmt)) {
            // first line should be var decl
            mir::VarDecl::Ptr frontier_var_decl = mir::to<mir::VarDecl>(first_stmt);

            // second line should be expr stmt
            if (mir::isa<mir::ExprStmt>(second_stmt)) {
                mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>(second_stmt);
                if (mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(expr_stmt->expr)) {
                    return true;
                }
            }
        }
        return false;
    }

    void PriorityFeaturesLower::LowerPriorityUpdateOperatorRewriter::visit(mir::Call::Ptr call) {
        auto call_args = call->args;
        if (call->name == "updatePriorityMin") {
            mir::PriorityUpdateOperatorMin::Ptr priority_update_min = std::make_shared<mir::PriorityUpdateOperatorMin>();

            priority_update_min->name = call->name;
            priority_update_min->args = call->args;

            priority_update_min->priority_queue = call_args[0];
            priority_update_min->destination_node_id = call_args[1];
            priority_update_min->old_val = call_args[2];
            priority_update_min->new_val = call_args[3];

            mir::VarDecl::Ptr priority_queue_decl = mir_context_->getPriorityQueueDecl();
            mir::ScalarType::Ptr priority_value_type
                    = (mir::to<mir::PriorityQueueType>(priority_queue_decl->type))->priority_type;

            priority_update_min->generic_type = priority_value_type;

            node = priority_update_min;
        } else if (call->name == "updatePrioritySum") {

            if (mir_context_->priority_update_type == mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate) {
                // we don't need to do any specialization for ConstSumReduceBeforePriorityUpdate
                node = call;
            } else {
                mir::PriorityUpdateOperatorSum::Ptr priority_udpate_sum = std::make_shared<mir::PriorityUpdateOperatorSum>();
                priority_udpate_sum->args = call->args;
                priority_udpate_sum->name = call->name;
                priority_udpate_sum->priority_queue = call_args[0];
                priority_udpate_sum->destination_node_id = call_args[1];
                priority_udpate_sum->delta = call_args[2];
                priority_udpate_sum->minimum_val = call_args[3];
                node = priority_udpate_sum;
            }
        } else {
            node = call;
        }

        //node = call;
    }

    //void LowerUpdatePriorityEdgeSetApplyExpr::visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr expr) {
    void PriorityFeaturesLower::LowerUpdatePriorityEdgeSetApplyExpr::visit(mir::ExprStmt::Ptr stmt) {

        node = stmt;
        if (!mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(stmt->expr))
            return;
        if (stmt->stmt_label != "") {
            label_scope_.scope(stmt->stmt_label);
        }
        auto expr = mir::to<mir::UpdatePriorityEdgeSetApplyExpr>(stmt->expr);
        if (schedule_ != nullptr && schedule_->apply_schedules != nullptr) {
            auto current_label = label_scope_.getCurrentScope();
            auto apply_schedule = schedule_->apply_schedules->find(current_label);
            if (apply_schedule != schedule_->apply_schedules->end()) { //a schedule is found
                if (apply_schedule->second.priority_update_type ==
                    ApplySchedule::PriorityUpdateType::CONST_SUM_REDUCTION_BEFORE_UPDATE) {
                    auto new_expr = std::make_shared<mir::UpdatePriorityEdgeCountEdgeSetApplyExpr>();
                    new_expr->copyFrom(expr);
                    //new_expr->lambda_name = "place_holder_lamda";
                    new_expr->moved_object_name = "moved_object_" + mir_context_->getUniqueNameCounterString();
                    new_expr->priority_queue_name = mir_context_->getPriorityQueueDecl()->name;
                    stmt->expr = new_expr;


                    mir::UpdatePriorityUpdateBucketsCall::Ptr update_call = std::make_shared<mir::UpdatePriorityUpdateBucketsCall>();
                    update_call->lambda_name = new_expr->moved_object_name + ".get_fn_repr()";
                    update_call->modified_vertexsubset_name = new_expr->moved_object_name;
                    update_call->priority_queue_name = new_expr->priority_queue_name;
                    update_call->priority_update_type = mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate;

                    mir::StmtBlock::Ptr stmt_block = std::make_shared<mir::StmtBlock>();
                    stmt_block->insertStmtEnd(stmt);
                    stmt_block->insertStmtEnd(update_call);

                    node = stmt_block;

                } else if (apply_schedule->second.priority_update_type ==
                           ApplySchedule::PriorityUpdateType::REDUCTION_BEFORE_UPDATE){
                    lowerReduceBeforeUpdateEdgeSetApply(stmt);
                }
            }
        } else {
            //The default priority update schedule is lower into ReduceBeforeUpdate
            lowerReduceBeforeUpdateEdgeSetApply(stmt);
        }
        if (stmt->stmt_label != "") {
            label_scope_.scope(stmt->stmt_label);
        }
    }

    void PriorityFeaturesLower::LowerUpdatePriorityEdgeSetApplyExpr::lowerReduceBeforeUpdateEdgeSetApply(
            mir::ExprStmt::Ptr stmt) {
        // convert it back to a normal edgeset apply expression but with a tracking variable
        assert (mir::isa<mir::UpdatePriorityEdgeSetApplyExpr>(stmt->expr));
        mir::EdgeSetApplyExpr::Ptr regular_edgeset_apply_expr = std::make_shared<mir::EdgeSetApplyExpr>();
        regular_edgeset_apply_expr->copyEdgesetApply(stmt->expr);
        regular_edgeset_apply_expr->enable_deduplication = true;
        regular_edgeset_apply_expr->tracking_field = mir_context_->getPriorityVectorName();

        // Create a new VarDecl statement that returns a frontier
        mir::VarDecl::Ptr new_var_decl = std::make_shared<mir::VarDecl>();
        std::string modified_vertexset_name = "modified_vertexsubset" + mir_context_->getUniqueNameCounterString();
        new_var_decl->initVal = regular_edgeset_apply_expr;
        new_var_decl->name = modified_vertexset_name;
        mir::VertexSetType::Ptr vertexset_type = std::make_shared<mir::VertexSetType>();
        vertexset_type->priority_update_type= mir::ReduceBeforePriorityUpdate;
        mir::VarExpr::Ptr edgeset_target = mir::to<mir::VarExpr>(regular_edgeset_apply_expr->target);
        // The new variable declaration inherits the label from the original expr stmt (such as "s1")
        // This allows the schedules to be assigned to the newly created vardecl stmt.
        new_var_decl->stmt_label = stmt->stmt_label;

        auto var = mir_context_->getSymbol(edgeset_target->var.getName());
        auto type = var.getType();
        mir::EdgeSetType::Ptr edgeset_type = mir::to<mir::EdgeSetType>(type);
        vertexset_type->element = edgeset_type->vertex_element_type_list->at(0);
        new_var_decl->type = vertexset_type;

        // Create a new call to update the buckets with the frontier
        mir::UpdatePriorityUpdateBucketsCall::Ptr update_call = std::make_shared<mir::UpdatePriorityUpdateBucketsCall>();
        update_call->lambda_name = modified_vertexset_name;
        update_call->modified_vertexsubset_name = modified_vertexset_name;
        update_call->priority_queue_name = mir_context_->getPriorityQueueDecl()->name;
        update_call->priority_update_type = mir::PriorityUpdateType::ReduceBeforePriorityUpdate;
        update_call->delta = mir_context_->delta_;
        update_call->nodes_init_in_bucket = mir_context_->nodes_init_in_buckets;
        mir::StmtBlock::Ptr stmt_block = std::make_shared<mir::StmtBlock>();
        stmt_block->insertStmtEnd(new_var_decl);
        stmt_block->insertStmtEnd(update_call);

        node = stmt_block;


    }


    void PriorityFeaturesLower::LowerReduceBeforePriorityUpdate::visit(mir::Call::Ptr expr) {

        // Locates the dequeue_ready_set() call and replace with
        if (expr->name == "dequeue_ready_set"
        && mir_context_->priority_update_type == mir::ReduceBeforePriorityUpdate){
            expr->name = "getBucketWithGraphItVertexSubset";
            node = expr;
        } else {
            node = expr;
        }
    }

}
