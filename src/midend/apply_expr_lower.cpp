//
// Created by Yunming Zhang on 5/30/17.
//

#include <graphit/midend/apply_expr_lower.h>

namespace graphit {
    //lowers vertexset apply and edgeset apply expressions according to schedules
    void ApplyExprLower::lower() {
        auto lower_apply_expr = LowerApplyExpr(schedule_, mir_context_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions) {
            lower_apply_expr.rewrite(function);
        }
    }

    void ApplyExprLower::LowerApplyExpr::visit(mir::VertexSetApplyExpr::Ptr vertexset_apply) {
        //default the vertexset apply expression to parallel (serial needs to be manually specified)
        vertexset_apply->is_parallel = true;

        if (schedule_ != nullptr && schedule_->apply_schedules != nullptr) {
            // We assume that there is only one apply in each statement
            auto current_scope_name = label_scope_.getCurrentScope();
            auto apply_schedule = schedule_->apply_schedules->find(current_scope_name);
            if (apply_schedule != schedule_->apply_schedules->end()) {
                //if a schedule for the statement has been found
                //Check to see if it is parallel or serial
                if (apply_schedule->second.parallel_type == ApplySchedule::ParType::Parallel) {
                    vertexset_apply->is_parallel = true;
                } else if (apply_schedule->second.parallel_type == ApplySchedule::ParType::Serial) {
                    vertexset_apply->is_parallel = false;
                }
            }
        }

        node = vertexset_apply;
    }

    void ApplyExprLower::LowerApplyExpr::visit(mir::EdgeSetApplyExpr::Ptr edgeset_apply) {
        // use the target var expressionto figure out the edgeset type
        mir::VarExpr::Ptr edgeset_expr = mir::to<mir::VarExpr>(edgeset_apply->target);
        //mir::VarDecl::Ptr edgeset_var_decl = mir_context_->getConstEdgeSetByName(edgeset_expr->var.getName());
        mir::EdgeSetType::Ptr edgeset_type = mir::to<mir::EdgeSetType>(edgeset_expr->var.getType());
        assert(edgeset_type->vertex_element_type_list->size() == 2);
        mir::ElementType::Ptr dst_vertex_type = (*(edgeset_type->vertex_element_type_list))[1];
        auto dst_vertices_range_expr = mir_context_->getElementCount(dst_vertex_type);

        if (edgeset_type->weight_type != nullptr) {
            edgeset_apply->is_weighted = true;
        }

        // check if the schedule contains entry for the current edgeset apply expressions
        if (schedule_ != nullptr && schedule_->apply_schedules != nullptr) {

            // We assume that there is only one apply in each statement
            auto current_scope_name = label_scope_.getCurrentScope();
            auto apply_schedule = schedule_->apply_schedules->find(current_scope_name);

            if (apply_schedule != schedule_->apply_schedules->end()) {
                // a schedule is found

                //First figure out the direction, and allocate the relevant edgeset expression
                if (apply_schedule->second.direction_type == ApplySchedule::DirectionType::PUSH) {
                    node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
                } else if (apply_schedule->second.direction_type == ApplySchedule::DirectionType::PULL) {
                    //Pull
                    node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
                } else if (apply_schedule->second.direction_type ==
                           ApplySchedule::DirectionType::HYBRID_DENSE_FORWARD) {
                    //Hybrid dense forward (switching betweeen push and dense forward push)
                    node = std::make_shared<mir::HybridDenseForwardEdgeSetApplyExpr>(edgeset_apply);
                } else if (apply_schedule->second.direction_type == ApplySchedule::DirectionType::HYBRID_DENSE) {
                    //Hybrid dense (switching betweeen push and pull)
                    auto hybrid_dense_edgeset_apply = std::make_shared<mir::HybridDenseEdgeSetApplyExpr>(edgeset_apply);
                    //clone the function delcaration for push, use the original func for pull
                    auto pull_apply_func_decl = mir_context_->getFunction(edgeset_apply->input_function->function_name->name);
                    mir::FuncDecl::Ptr push_apply_func_decl = pull_apply_func_decl->clone<mir::FuncDecl>();
                    push_apply_func_decl->name = push_apply_func_decl->name + "_push_ver";
                    hybrid_dense_edgeset_apply->push_function_ = std::make_shared<mir::FuncExpr>();
                    hybrid_dense_edgeset_apply->push_function_->function_name = std::make_shared<mir::IdentDecl>();
                    hybrid_dense_edgeset_apply->push_function_->function_name->name = push_apply_func_decl->name;
                    //TODO is this correct assumption to make?
                    hybrid_dense_edgeset_apply->push_function_->functorArgs = edgeset_apply->input_function->functorArgs;
                    //insert into MIR context
                    mir_context_->addFunctionFront(push_apply_func_decl);

                    node = hybrid_dense_edgeset_apply;
                }

                // No longer need this check as we moved the check to high-level scheduling API
                // We use negative integers between -1 and -10 to denote argv numbers
                // it can't be 0 as well, which indicates that this schedule is not needed
                if (apply_schedule->second.num_segment > -10 && apply_schedule->second.num_segment != 0) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->scope_label_name = apply_schedule->second.scope_label_name;
                    mir_context_->edgeset_to_label_to_num_segment[edgeset_expr->var.getName()][apply_schedule->second.scope_label_name] =
                            apply_schedule->second.num_segment;
                }

                //Check to see if it is parallel or serial
                if (apply_schedule->second.parallel_type == ApplySchedule::ParType::Parallel) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->is_parallel = true;
                } else if (apply_schedule->second.parallel_type == ApplySchedule::ParType::Serial) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->is_parallel = false;
                }

                if (apply_schedule->second.opt == ApplySchedule::OtherOpt::SLIDING_QUEUE) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->use_sliding_queue = true;
                }

                if (apply_schedule->second.pull_frontier_type == ApplySchedule::PullFrontierType ::BITVECTOR) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->use_pull_frontier_bitvector = true;
                }

                if (apply_schedule->second.pull_load_balance_type == ApplySchedule::PullLoadBalance::EDGE_BASED){
                    mir::to<mir::EdgeSetApplyExpr>(node)->use_pull_edge_based_load_balance = true;
                    if (apply_schedule->second.pull_load_balance_edge_grain_size > 0){
                        mir::to<mir::EdgeSetApplyExpr>(node)->pull_edge_based_load_balance_grain_size
                                = apply_schedule->second.pull_load_balance_edge_grain_size;
                    }
                }


                mir::to<mir::EdgeSetApplyExpr>(node)->grain_size = apply_schedule->second.grain_size;


                //if this is applyModified with a tracking field
                if (edgeset_apply->tracking_field != "") {
                    // only enable deduplication when the argument to ApplyModified is True (disable deduplication), or the user manually set disable
                    if (edgeset_apply->enable_deduplication && apply_schedule->second.deduplication_type == ApplySchedule::DeduplicationType::Enable) {
                        //only enable deduplication if there is needed for tracking
                        mir::to<mir::EdgeSetApplyExpr>(node)->enable_deduplication = true;
                    }
                } else {
                    mir::to<mir::EdgeSetApplyExpr>(node)->enable_deduplication = false;
                }
            } else {
                //There is a schedule, but nothing is specified for the current apply
                node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
                return;
            }

            //mir::to<mir::EdgeSetApplyExpr>(node)->grain_size = stuff->second->
            return;
        } else {
            //setting the default direction to push if no schedule is specified
            node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
            return;
        }
    }

}
