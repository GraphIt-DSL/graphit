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
    void ApplyExprLower::LowerApplyExpr::visit(mir::StmtBlock::Ptr stmt_block) {
	std::vector<mir::Stmt::Ptr> new_stmts;
	for (auto stmt: *(stmt_block->stmts)) {
		new_stmts.push_back(rewrite<mir::Stmt>(stmt));
		if (insert_after_stmt != nullptr)
			new_stmts.push_back(insert_after_stmt);
		insert_after_stmt = nullptr;	
	}
	* (stmt_block->stmts) = new_stmts;
	node = stmt_block;
    }
    void ApplyExprLower::LowerApplyExpr::visit(mir::VarDecl::Ptr var_decl) {
	MIRRewriter::visit(var_decl);
	var_decl = mir::to<mir::VarDecl>(node);
	if (mir::isa<mir::EdgeSetApplyExpr> (var_decl->initVal)) {
		mir::EdgeSetApplyExpr::Ptr edgeset_apply = mir::to<mir::EdgeSetApplyExpr>(var_decl->initVal);
		
		if (edgeset_apply->applied_schedule.deduplication == fir::gpu_schedule::SimpleGPUSchedule::deduplication_type::DEDUP_ENABLED && edgeset_apply->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
			mir::VertexSetDedupExpr::Ptr dedup_expr = std::make_shared<mir::VertexSetDedupExpr>();
			mir::ExprStmt::Ptr expr_stmt = std::make_shared<mir::ExprStmt>();
			mir::Var var(var_decl->name, var_decl->type);
			mir::VarExpr::Ptr var_expr = std::make_shared<mir::VarExpr>();
			var_expr->var = var;
			dedup_expr->target = var_expr;
			
			expr_stmt->expr = dedup_expr;
			insert_after_stmt = expr_stmt;
		}
	}
	node = var_decl;
    }    
    void ApplyExprLower::LowerApplyExpr::visit(mir::AssignStmt::Ptr assign_stmt) {
        MIRRewriter::visit(assign_stmt);
	assign_stmt = mir::to<mir::AssignStmt>(node);
	node = assign_stmt;
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



	// First check if the program has a GPU Schedule, if yes, the defaults are different
	if (schedule_ != nullptr && !schedule_->apply_gpu_schedules.empty()) {
		// Always parallelize all operators for GPU schedules
		edgeset_apply->is_parallel = true;
		// Check if there is a GPU schedule attached to this statement - 
            	auto current_scope_name = label_scope_.getCurrentScope();
		auto apply_schedule_iter = schedule_->apply_gpu_schedules.find(current_scope_name);
		if (apply_schedule_iter != schedule_->apply_gpu_schedules.end()) {
			auto apply_schedule = apply_schedule_iter->second;
			if (dynamic_cast<fir::gpu_schedule::SimpleGPUSchedule*>(apply_schedule) != nullptr) {	
				edgeset_apply->applied_schedule = *dynamic_cast<fir::gpu_schedule::SimpleGPUSchedule*>(apply_schedule);
			}
			if (edgeset_apply->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PUSH)
				node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
			else if (edgeset_apply->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL)
				node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
			else 
				assert(false && "Invalid option for direction\n");
						
		} else {
			// No schedule is attached, lower using default schedule	
			node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);			
		}
		return;
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
                    auto pull_apply_func_decl = mir_context_->getFunction(edgeset_apply->input_function_name);
                    mir::FuncDecl::Ptr push_apply_func_decl = pull_apply_func_decl->clone<mir::FuncDecl>();
                    push_apply_func_decl->name = push_apply_func_decl->name + "_push_ver";
                    hybrid_dense_edgeset_apply->push_function_ = push_apply_func_decl->name;
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

            return;
        } else {
            //setting the default direction to push if no schedule is specified
            node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
            return;
        }
    }

}
