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
		while (insert_after_stmt != nullptr) {
			auto temp = insert_after_stmt;
			insert_after_stmt = nullptr;	
			temp = rewrite<mir::Stmt>(temp);
			new_stmts.push_back(temp);
		}
	}
	* (stmt_block->stmts) = new_stmts;
	node = stmt_block;
    }
    void ApplyExprLower::LowerApplyExpr::visit(mir::VarDecl::Ptr var_decl) {
	if (schedule_ != nullptr && !schedule_->apply_gpu_schedules.empty()) {
		if (mir::isa<mir::EdgeSetApplyExpr> (var_decl->initVal) || mir::isa<mir::VertexSetWhereExpr>(var_decl->initVal)) {
			auto init_val = var_decl->initVal;
			var_decl->initVal = nullptr;
			mir::AssignStmt::Ptr assign_stmt = std::make_shared<mir::AssignStmt>();
			assign_stmt->expr = init_val;
			mir::VarExpr::Ptr var_expr = std::make_shared<mir::VarExpr>();
			mir::Var var (var_decl->name, var_decl->type);
			var_expr->var = var;
			assign_stmt->lhs = var_expr;
			assign_stmt->stmt_label = var_decl->stmt_label;
			insert_after_stmt = assign_stmt;
			node = var_decl;
			return;	
		} 
	}
	MIRRewriter::visit(var_decl);
	var_decl = mir::to<mir::VarDecl>(node);
	node = var_decl;
    }    
    void ApplyExprLower::LowerApplyExpr::visit(mir::AssignStmt::Ptr assign_stmt) {
	
        if (assign_stmt->stmt_label != "") {
                label_scope_.scope(assign_stmt->stmt_label);
        }
		
	// Check for Hybrid stmt
	if (mir::isa<mir::EdgeSetApplyExpr> (assign_stmt->expr)) {
		mir::EdgeSetApplyExpr::Ptr edgeset_apply = mir::to<mir::EdgeSetApplyExpr>(assign_stmt->expr);
		if (schedule_ != nullptr && !schedule_->apply_gpu_schedules.empty()) {
			auto current_scope_name = label_scope_.getCurrentScope();
			auto apply_schedule_iter = schedule_->apply_gpu_schedules.find(current_scope_name);
			if (apply_schedule_iter != schedule_->apply_gpu_schedules.end()) {
				auto apply_schedule = apply_schedule_iter->second;
				if (dynamic_cast<fir::gpu_schedule::HybridGPUSchedule*>(apply_schedule) != nullptr) {	
					fir::gpu_schedule::HybridGPUSchedule *hybrid_schedule = dynamic_cast<fir::gpu_schedule::HybridGPUSchedule*>(apply_schedule);	
					// This EdgeSetApply has a Hybrid Schedule attached to it
					// Create the first Stmt block
					mir::StmtBlock::Ptr stmt_block_1 = std::make_shared<mir::StmtBlock>();	
					mir::AssignStmt::Ptr stmt1 = std::make_shared<mir::AssignStmt>();
					stmt1->lhs = assign_stmt->lhs;
					stmt1->expr = assign_stmt->expr;
					stmt1->stmt_label = "hybrid1";	
					stmt_block_1->insertStmtEnd(stmt1);
					fir::gpu_schedule::SimpleGPUSchedule * schedule1 = new fir::gpu_schedule::SimpleGPUSchedule();
					*schedule1 = hybrid_schedule->s1;
					schedule_->apply_gpu_schedules[current_scope_name + ":hybrid1"] = schedule1;
					stmt_block_1 = rewrite<mir::StmtBlock>(stmt_block_1);
					
					// Now create the second Stmt block
				        auto func_decl = mir_context_->getFunction(edgeset_apply->input_function->function_name->name);
				        mir::FuncDecl::Ptr func_decl_v2 = func_decl->clone<mir::FuncDecl>();
				        func_decl_v2->name = func_decl->name + "_v2"; 
				        mir_context_->addFunctionFront(func_decl_v2);
					mir::StmtBlock::Ptr stmt_block_2 = std::make_shared<mir::StmtBlock>();
					mir::AssignStmt::Ptr stmt2 = std::make_shared<mir::AssignStmt>();
					stmt2->lhs = assign_stmt->lhs;
					stmt2->expr = assign_stmt->expr;
						
					mir::FuncExpr::Ptr new_func_expr = std::make_shared<mir::FuncExpr>();
					new_func_expr->function_name = std::make_shared<mir::IdentDecl>();
					new_func_expr->function_name->name = func_decl_v2->name;


					mir::to<mir::EdgeSetApplyExpr>(stmt2->expr)->input_function= new_func_expr;
					stmt2->stmt_label = "hybrid2";
					stmt_block_2->insertStmtEnd(stmt2);
					fir::gpu_schedule::SimpleGPUSchedule * schedule2 = new fir::gpu_schedule::SimpleGPUSchedule();
					*schedule2 = hybrid_schedule->s2;
					schedule_->apply_gpu_schedules[current_scope_name + ":hybrid2"] = schedule2;
					stmt_block_2 = rewrite<mir::StmtBlock>(stmt_block_2);
					
					// Finally create a hybrid statement and replace - 
					mir::HybridGPUStmt::Ptr hybrid_node = std::make_shared<mir::HybridGPUStmt>();
					hybrid_node->stmt1 = stmt_block_1;
					hybrid_node->stmt2 = stmt_block_2;
					hybrid_node->threshold = hybrid_schedule->threshold;
					hybrid_node->argv_index = hybrid_schedule->argv_index;
					hybrid_node->criteria = hybrid_schedule->_hybrid_criteria;
					if (hybrid_node->criteria == fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria::INPUT_VERTEXSET_SIZE && edgeset_apply->from_func->function_name->name != "") {
						hybrid_node->input_frontier_name = edgeset_apply->from_func->function_name->name;	
					} else {
						assert(false && "Invalid criteria for Hybrid Node\n");
					}
					
					node = hybrid_node;
					mir_context_->hybrid_gpu_stmts.push_back(hybrid_node);
					if (assign_stmt->stmt_label != "") {
						label_scope_.unscope();
					}
					return;
								
				}
			}
		}
	}
        if (assign_stmt->stmt_label != "") {
                label_scope_.unscope();
        }


        MIRRewriter::visit(assign_stmt);
	assign_stmt = mir::to<mir::AssignStmt>(node);
	if (mir::isa<mir::EdgeSetApplyExpr> (assign_stmt->expr)) {
		mir::EdgeSetApplyExpr::Ptr edgeset_apply = mir::to<mir::EdgeSetApplyExpr>(assign_stmt->expr);
		if (schedule_ != nullptr && !schedule_->apply_gpu_schedules.empty() && edgeset_apply->enable_deduplication == true && edgeset_apply->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
			if (edgeset_apply->applied_schedule.deduplication_strategy == fir::gpu_schedule::SimpleGPUSchedule::deduplication_strategy_type::DEDUP_FUSED) {
				edgeset_apply->fused_dedup = true;
				edgeset_apply->fused_dedup_perfect = true;
			} else {
				mir::VertexSetDedupExpr::Ptr dedup_expr = std::make_shared<mir::VertexSetDedupExpr>();
				mir::ExprStmt::Ptr expr_stmt = std::make_shared<mir::ExprStmt>();
				dedup_expr->target = assign_stmt->lhs;	
				expr_stmt->expr = dedup_expr;
				insert_after_stmt = expr_stmt;
				dedup_expr->perfect_dedup = true;
				edgeset_apply->fused_dedup = false;
			}
		} else if (schedule_ != nullptr && !schedule_->apply_gpu_schedules.empty() && edgeset_apply->applied_schedule.deduplication == fir::gpu_schedule::SimpleGPUSchedule::deduplication_type::DEDUP_ENABLED && edgeset_apply->applied_schedule.frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
			if (edgeset_apply->applied_schedule.deduplication_strategy == fir::gpu_schedule::SimpleGPUSchedule::deduplication_strategy_type::DEDUP_FUSED) {
				edgeset_apply->fused_dedup = true;
				edgeset_apply->fused_dedup_perfect = false;
			} else {
				mir::VertexSetDedupExpr::Ptr dedup_expr = std::make_shared<mir::VertexSetDedupExpr>();
				mir::ExprStmt::Ptr expr_stmt = std::make_shared<mir::ExprStmt>();
				dedup_expr->target = assign_stmt->lhs;	
				expr_stmt->expr = dedup_expr;
				insert_after_stmt = expr_stmt;
				dedup_expr->perfect_dedup = false;
				edgeset_apply->fused_dedup = false;
			}
		}
	}
	node = assign_stmt;
    }
    void ApplyExprLower::LowerApplyExpr::visit(mir::ExprStmt::Ptr expr_stmt) {
        if (expr_stmt->stmt_label != "") {
                label_scope_.scope(expr_stmt->stmt_label);
        }
	if (mir::isa<mir::EdgeSetApplyExpr> (expr_stmt->expr)) {
		mir::EdgeSetApplyExpr::Ptr edgeset_apply = mir::to<mir::EdgeSetApplyExpr>(expr_stmt->expr);
		if (schedule_ != nullptr && !schedule_->apply_gpu_schedules.empty()) {
			auto current_scope_name = label_scope_.getCurrentScope();
			auto apply_schedule_iter = schedule_->apply_gpu_schedules.find(current_scope_name);
			if (apply_schedule_iter != schedule_->apply_gpu_schedules.end()) {
				auto apply_schedule = apply_schedule_iter->second;
				if (dynamic_cast<fir::gpu_schedule::HybridGPUSchedule*>(apply_schedule) != nullptr) {	
					fir::gpu_schedule::HybridGPUSchedule *hybrid_schedule = dynamic_cast<fir::gpu_schedule::HybridGPUSchedule*>(apply_schedule);	
					// This EdgeSetApply has a Hybrid Schedule attached to it
					// Create the first Stmt block
					mir::StmtBlock::Ptr stmt_block_1 = std::make_shared<mir::StmtBlock>();	
					mir::ExprStmt::Ptr stmt1 = std::make_shared<mir::ExprStmt>();
					stmt1->expr = expr_stmt->expr;
					stmt1->stmt_label = "hybrid1";	
					stmt_block_1->insertStmtEnd(stmt1);
					fir::gpu_schedule::SimpleGPUSchedule * schedule1 = new fir::gpu_schedule::SimpleGPUSchedule();
					*schedule1 = hybrid_schedule->s1;
					schedule_->apply_gpu_schedules[current_scope_name + ":hybrid1"] = schedule1;
					stmt_block_1 = rewrite<mir::StmtBlock>(stmt_block_1);
					
					// Now create the second Stmt block
				        auto func_decl = mir_context_->getFunction(edgeset_apply->input_function->function_name->name);
				        mir::FuncDecl::Ptr func_decl_v2 = func_decl->clone<mir::FuncDecl>();
				        func_decl_v2->name = func_decl->name + "_v2"; 
				        mir_context_->addFunctionFront(func_decl_v2);
					mir::StmtBlock::Ptr stmt_block_2 = std::make_shared<mir::StmtBlock>();
					mir::ExprStmt::Ptr stmt2 = std::make_shared<mir::ExprStmt>();
					stmt2->expr = expr_stmt->expr;

					mir::FuncExpr::Ptr new_func_expr = std::make_shared<mir::FuncExpr>();
					new_func_expr->function_name = std::make_shared<mir::IdentDecl>();
					new_func_expr->function_name->name = func_decl_v2->name;

					mir::to<mir::EdgeSetApplyExpr>(stmt2->expr)->input_function = new_func_expr;
					stmt2->stmt_label = "hybrid2";
					stmt_block_2->insertStmtEnd(stmt2);
					fir::gpu_schedule::SimpleGPUSchedule * schedule2 = new fir::gpu_schedule::SimpleGPUSchedule();
					*schedule2 = hybrid_schedule->s2;
					schedule_->apply_gpu_schedules[current_scope_name + ":hybrid2"] = schedule2;
					stmt_block_2 = rewrite<mir::StmtBlock>(stmt_block_2);
					
					// Finally create a hybrid statement and replace - 
					mir::HybridGPUStmt::Ptr hybrid_node = std::make_shared<mir::HybridGPUStmt>();
					hybrid_node->stmt1 = stmt_block_1;
					hybrid_node->stmt2 = stmt_block_2;
					hybrid_node->threshold = hybrid_schedule->threshold;
					hybrid_node->argv_index = hybrid_schedule->argv_index;
					hybrid_node->criteria = hybrid_schedule->_hybrid_criteria;
					if (hybrid_node->criteria == fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria::INPUT_VERTEXSET_SIZE && edgeset_apply->from_func->function_name->name != "") {
						hybrid_node->input_frontier_name = edgeset_apply->from_func->function_name->name;	
					} else {
						assert(false && "Invalid criteria for Hybrid Node\n");
					}
					
					node = hybrid_node;
					mir_context_->hybrid_gpu_stmts.push_back(hybrid_node);
					if (expr_stmt->stmt_label != "") {
						label_scope_.unscope();
					}
					return;
					
				}
			}
		}
	}
        if (expr_stmt->stmt_label != "") {
                label_scope_.unscope();
        }
        MIRRewriter::visit(expr_stmt);
	node = expr_stmt;
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
		if (edgeset_apply->tracking_field != "")
			edgeset_apply->requires_output = true;
		// Check if there is a GPU schedule attached to this statement - 
            	auto current_scope_name = label_scope_.getCurrentScope();
		auto apply_schedule_iter = schedule_->apply_gpu_schedules.find(current_scope_name);
		if (apply_schedule_iter != schedule_->apply_gpu_schedules.end()) {
			auto apply_schedule = apply_schedule_iter->second;
			if (dynamic_cast<fir::gpu_schedule::SimpleGPUSchedule*>(apply_schedule) != nullptr) {	
				edgeset_apply->applied_schedule = *dynamic_cast<fir::gpu_schedule::SimpleGPUSchedule*>(apply_schedule);
			} else {
				assert(false && "Schedule applied to EdgeSetApply must be a Simple Schedule");
			}
			if (edgeset_apply->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PUSH)
				node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
			else if (edgeset_apply->applied_schedule.direction == fir::gpu_schedule::SimpleGPUSchedule::direction_type::DIR_PULL) {
				node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
				mir_context_->graphs_with_transpose[mir::to<mir::VarExpr>(edgeset_apply->target)->var.getName()] = true;
			} else 
				assert(false && "Invalid option for direction\n");
			
			if (edgeset_apply->applied_schedule.load_balancing == fir::gpu_schedule::SimpleGPUSchedule::load_balancing_type::EDGE_ONLY && edgeset_apply->applied_schedule.edge_blocking == fir::gpu_schedule::SimpleGPUSchedule::edge_blocking_type::BLOCKED) {
				mir_context_->graphs_with_blocking[mir::to<mir::VarExpr>(edgeset_apply->target)->var.getName()] = edgeset_apply->applied_schedule.edge_blocking_size;
			}
						
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
