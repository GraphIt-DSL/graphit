//
// Created by Yunming Zhang on 5/30/17.
//

#include <graphit/midend/apply_expr_lower.h>

namespace graphit {

    using fir::abstract_schedule::ScheduleObject;
    using fir::abstract_schedule::SimpleScheduleObject;
    using fir::cpu_schedule::SimpleCPUScheduleObject;
    using fir::gpu_schedule::SimpleGPUSchedule;

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
        vertexset_apply->setMetadata<bool>("is_parallel", true);

        if (schedule_->backend_identifier == Schedule::BackendID::CPU) {
            // We assume that there is only one apply in each statement
            if (vertexset_apply->hasApplySchedule()) {
              auto apply_schedule = vertexset_apply->getApplySchedule<SimpleScheduleObject>();
              vertexset_apply->setMetadata<bool>("is_parallel",
                  !(apply_schedule->to<SimpleCPUScheduleObject>()->getCPUParallelizationType()
                      == SimpleCPUScheduleObject::CPUParallelType::SERIAL));
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
	if (schedule_->backend_identifier == Schedule::BackendID::GPU) {
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

			// copy over any schedule from the var_decl to the new assign stmt.
			if (var_decl->hasApplySchedule()) {
			  assign_stmt->setMetadata<ScheduleObject::Ptr>("apply_schedule", var_decl->getApplySchedule());
			}
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
		if (schedule_->backend_identifier == Schedule::BackendID::GPU) {
			if (assign_stmt->hasApplySchedule()) {
				auto apply_schedule = assign_stmt->getApplySchedule();
				if (apply_schedule->isComposite()) {
                    auto hybrid_schedule = apply_schedule->to<fir::gpu_schedule::HybridGPUSchedule>();
					// This EdgeSetApply has a Hybrid Schedule attached to it
					// Create the first Stmt block
					mir::StmtBlock::Ptr stmt_block_1 = std::make_shared<mir::StmtBlock>();	
					mir::AssignStmt::Ptr stmt1 = std::make_shared<mir::AssignStmt>();
					stmt1->lhs = assign_stmt->lhs;
					stmt1->expr = assign_stmt->expr;
					stmt1->stmt_label = "hybrid1";	
					stmt_block_1->insertStmtEnd(stmt1);
                    auto schedule1_copy = hybrid_schedule->getFirstScheduleObject()
                        ->self<fir::gpu_schedule::SimpleGPUSchedule>()->cloneSchedule();
					stmt1->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule1_copy);
                    assign_stmt->expr->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule1_copy);
					stmt_block_1 = rewrite<mir::StmtBlock>(stmt_block_1);
					
					// Now create the second Stmt block
				        auto func_decl = mir_context_->getFunction(edgeset_apply->input_function_name);
				        mir::FuncDecl::Ptr func_decl_v2 = func_decl->clone<mir::FuncDecl>();
				        func_decl_v2->name = func_decl->name + "_v2"; 
				        mir_context_->addFunctionFront(func_decl_v2);
					mir::StmtBlock::Ptr stmt_block_2 = std::make_shared<mir::StmtBlock>();
					mir::AssignStmt::Ptr stmt2 = std::make_shared<mir::AssignStmt>();
					stmt2->lhs = assign_stmt->lhs;
					mir::Expr::Ptr ptr_copy = assign_stmt->expr->clone<mir::Expr>();
					stmt2->expr = ptr_copy;

					mir::to<mir::EdgeSetApplyExpr>(stmt2->expr)->input_function_name = func_decl_v2->name;
					stmt2->stmt_label = "hybrid2";
					stmt_block_2->insertStmtEnd(stmt2);
                    auto schedule2_copy = hybrid_schedule->getSecondScheduleObject()
                        ->self<fir::gpu_schedule::SimpleGPUSchedule>()->cloneSchedule();
                    stmt2->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule2_copy);
                    ptr_copy->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule2_copy);
					stmt_block_2 = rewrite<mir::StmtBlock>(stmt_block_2);
					
					// Finally create a hybrid statement and replace - 
					mir::HybridGPUStmt::Ptr hybrid_node = std::make_shared<mir::HybridGPUStmt>();
					hybrid_node->stmt1 = stmt_block_1;
					hybrid_node->stmt2 = stmt_block_2;
					hybrid_node->setMetadata<float>("threshold", hybrid_schedule->threshold);
					hybrid_node->setMetadata<int32_t>("argv_index", hybrid_schedule->argv_index);
					hybrid_node->setMetadata<fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria>("criteria", hybrid_schedule->_hybrid_criteria);
					if (hybrid_node->getMetadata<fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria>("criteria") == fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria::INPUT_VERTEXSET_SIZE && edgeset_apply->from_func != "") {
						hybrid_node->setMetadata<std::string>("input_frontier_name", edgeset_apply->from_func);
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

		auto applied_schedule = std::make_shared<SimpleGPUSchedule>();
        if (schedule_->backend_identifier == Schedule::BackendID::GPU && edgeset_apply->hasApplySchedule()) {
          applied_schedule = edgeset_apply->getApplySchedule<SimpleGPUSchedule>();
        }
        edgeset_apply->setMetadata<bool>("fused_dedup", false);
        edgeset_apply->setMetadata<bool>("fused_dedup_perfect", false);
		if (schedule_->backend_identifier == Schedule::BackendID::GPU && edgeset_apply->getMetadata<bool>("enable_deduplication") && applied_schedule->frontier_creation == fir::gpu_schedule::SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
            if (applied_schedule->deduplication_strategy == SimpleGPUSchedule::deduplication_strategy_type::DEDUP_FUSED) {
				edgeset_apply->setMetadata<bool>("fused_dedup", true);
				edgeset_apply->setMetadata<bool>("fused_dedup_perfect", true);
			} else {
				mir::VertexSetDedupExpr::Ptr dedup_expr = std::make_shared<mir::VertexSetDedupExpr>();
				mir::ExprStmt::Ptr expr_stmt = std::make_shared<mir::ExprStmt>();
				dedup_expr->target = assign_stmt->lhs;	
				expr_stmt->expr = dedup_expr;
				insert_after_stmt = expr_stmt;
				dedup_expr->setMetadata<bool>("perfect_dedup", true);
				edgeset_apply->setMetadata<bool>("fused_dedup", false);
			}
		} else if (schedule_->backend_identifier == Schedule::BackendID::GPU && applied_schedule->deduplication == SimpleGPUSchedule::deduplication_type::DEDUP_ENABLED && applied_schedule->frontier_creation == SimpleGPUSchedule::frontier_creation_type::FRONTIER_FUSED) {
			if (applied_schedule->deduplication_strategy == SimpleGPUSchedule::deduplication_strategy_type::DEDUP_FUSED) {
				edgeset_apply->setMetadata<bool>("fused_dedup", true);
				edgeset_apply->setMetadata<bool>("fused_dedup_perfect", false);
			} else {
				mir::VertexSetDedupExpr::Ptr dedup_expr = std::make_shared<mir::VertexSetDedupExpr>();
				mir::ExprStmt::Ptr expr_stmt = std::make_shared<mir::ExprStmt>();
				dedup_expr->target = assign_stmt->lhs;	
				expr_stmt->expr = dedup_expr;
				insert_after_stmt = expr_stmt;
				dedup_expr->setMetadata<bool>("perfect_dedup", false);
				edgeset_apply->setMetadata<bool>("fused_dedup", false);
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
		if (schedule_->backend_identifier == Schedule::BackendID::GPU) {
			if (expr_stmt->hasApplySchedule()) {
              auto apply_schedule = expr_stmt->getApplySchedule();
              if (apply_schedule->isComposite()) {
                    fir::gpu_schedule::HybridGPUSchedule::Ptr hybrid_schedule = apply_schedule->to<fir::gpu_schedule::HybridGPUSchedule>();
					// This EdgeSetApply has a Hybrid Schedule attached to it
					// Create the first Stmt block
					mir::StmtBlock::Ptr stmt_block_1 = std::make_shared<mir::StmtBlock>();	
					mir::ExprStmt::Ptr stmt1 = std::make_shared<mir::ExprStmt>();
					stmt1->expr = expr_stmt->expr;
					stmt1->stmt_label = "hybrid1";	
					stmt_block_1->insertStmtEnd(stmt1);
                    auto schedule1_copy = hybrid_schedule->getFirstScheduleObject()
                        ->self<fir::gpu_schedule::SimpleGPUSchedule>()->cloneSchedule();
                    stmt1->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule1_copy);
                    expr_stmt->expr->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule1_copy);
					stmt_block_1 = rewrite<mir::StmtBlock>(stmt_block_1);
					
					// Now create the second Stmt block
				        auto func_decl = mir_context_->getFunction(edgeset_apply->input_function_name);
				        mir::FuncDecl::Ptr func_decl_v2 = func_decl->clone<mir::FuncDecl>();
				        func_decl_v2->name = func_decl->name + "_v2"; 
				        mir_context_->addFunctionFront(func_decl_v2);
					mir::StmtBlock::Ptr stmt_block_2 = std::make_shared<mir::StmtBlock>();
					mir::ExprStmt::Ptr stmt2 = std::make_shared<mir::ExprStmt>();
                    mir::Expr::Ptr ptr_copy = expr_stmt->expr->clone<mir::Expr>();
					stmt2->expr = ptr_copy;
					mir::to<mir::EdgeSetApplyExpr>(stmt2->expr)->input_function_name = func_decl_v2->name;
					stmt2->stmt_label = "hybrid2";
					stmt_block_2->insertStmtEnd(stmt2);
                    auto schedule2_copy = hybrid_schedule->getSecondScheduleObject()
                        ->self<fir::gpu_schedule::SimpleGPUSchedule>()->cloneSchedule();
                    stmt2->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule2_copy);
                    ptr_copy->setMetadata<ScheduleObject::Ptr>("apply_schedule", schedule2_copy);

					stmt_block_2 = rewrite<mir::StmtBlock>(stmt_block_2);
					
					// Finally create a hybrid statement and replace - 
					mir::HybridGPUStmt::Ptr hybrid_node = std::make_shared<mir::HybridGPUStmt>();
					hybrid_node->stmt1 = stmt_block_1;
					hybrid_node->stmt2 = stmt_block_2;
					hybrid_node->setMetadata<float>("threshold", hybrid_schedule->threshold);
					hybrid_node->setMetadata<int32_t>("argv_index", hybrid_schedule->argv_index);
					hybrid_node->setMetadata<fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria>("criteria", hybrid_schedule->_hybrid_criteria);
					if (hybrid_node->getMetadata<fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria>("criteria") == fir::gpu_schedule::HybridGPUSchedule::hybrid_criteria::INPUT_VERTEXSET_SIZE && edgeset_apply->from_func != "") {
						hybrid_node->setMetadata<std::string>("input_frontier_name", edgeset_apply->from_func);
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

        // GPU change tracking pass requires this flag to process ESAE's so attach it in Swarm case too.
    if (schedule_->backend_identifier == Schedule::BackendID::SWARM) {
      if (edgeset_apply->tracking_field != "")
        edgeset_apply->setMetadata<bool>("requires_output", true);
      node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
      return;
    }
	// First check if the program has a GPU Schedule, if yes, the defaults are different
	if (schedule_->backend_identifier == Schedule::BackendID::GPU ) {
		// Always parallelize all operators for GPU schedules
		edgeset_apply->setMetadata<bool>("is_parallel", true);
        if (edgeset_apply->tracking_field != "")
          edgeset_apply->setMetadata<bool>("requires_output", true);
		// Check if there is a GPU schedule attached to this statement - 
          if (edgeset_apply->hasApplySchedule() && schedule_->backend_identifier == Schedule::BackendID::GPU) {
            auto apply_schedule = edgeset_apply->getApplySchedule();

			if (apply_schedule->self<SimpleGPUSchedule>()->direction == SimpleGPUSchedule::direction_type::DIR_PUSH)
				node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
			else if (apply_schedule->self<SimpleGPUSchedule>()->direction == SimpleGPUSchedule::direction_type::DIR_PULL) {
				node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
				mir_context_->graphs_with_transpose[mir::to<mir::VarExpr>(edgeset_apply->target)->var.getName()] = true;
			} else 
				assert(false && "Invalid option for direction\n");
			
			if (apply_schedule->self<SimpleGPUSchedule>()->load_balancing == SimpleGPUSchedule::load_balancing_type::EDGE_ONLY &&
			    apply_schedule->self<SimpleGPUSchedule>()->edge_blocking == SimpleGPUSchedule::edge_blocking_type::BLOCKED) {
				mir_context_->graphs_with_blocking[mir::to<mir::VarExpr>(edgeset_apply->target)->var.getName()] =
				    apply_schedule->self<SimpleGPUSchedule>()->edge_blocking_size;
			}
						
		} else {
			// No schedule is attached, lower using default schedule	
			node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);			
		}
		return;
	}

        // check if the schedule contains entry for the current edgeset apply expressions

        if (schedule_->backend_identifier == Schedule::BackendID::CPU) {

            // We assume that there is only one apply in each statement
              if (edgeset_apply->hasApplySchedule()) {
                auto apply_schedule = edgeset_apply->getApplySchedule();
                // a schedule is found

                //First figure out the direction, and allocate the relevant edgeset expression
                if (!apply_schedule->isComposite()) {
                  auto simple_schedule = apply_schedule->self<SimpleScheduleObject>();

                  if (simple_schedule->getDirection() == SimpleScheduleObject::Direction::PUSH) {
                    node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
                  } else if (simple_schedule->getDirection() == SimpleScheduleObject::Direction::PULL) {
                    //Pull
                    node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
                  }
                }
                else {
                  auto hybrid_schedule = apply_schedule->self<fir::abstract_schedule::CompositeScheduleObject>();
                  // Slightly hacky, but basically infer whether it's HYBRID_DENSE_FORWARD from whether one of the
                  // schedules is a Pull schedule. Both schedules in HYBRID_DENSE_FORWARD are push schedules.
                  if (hybrid_schedule->getFirstScheduleObject()->self<SimpleScheduleObject>()->getDirection()
                      != SimpleScheduleObject::Direction::PULL &&
                      hybrid_schedule->getSecondScheduleObject()->self<SimpleScheduleObject>()->getDirection()
                          != SimpleScheduleObject::Direction::PULL) {
                    //Hybrid dense forward (switching between push and dense forward push)
                    node = std::make_shared<mir::HybridDenseForwardEdgeSetApplyExpr>(edgeset_apply);
                  } else {
                    //Hybrid dense (switching between push and pull)
                    auto hybrid_dense_edgeset_apply =
                        std::make_shared<mir::HybridDenseEdgeSetApplyExpr>(edgeset_apply);
                    //clone the function declaration for push, use the original func for pull
                    auto pull_apply_func_decl = mir_context_->getFunction(edgeset_apply->input_function_name);
                    mir::FuncDecl::Ptr push_apply_func_decl = pull_apply_func_decl->clone<mir::FuncDecl>();
                    push_apply_func_decl->name = push_apply_func_decl->name + "_push_ver";
                    hybrid_dense_edgeset_apply->setMetadata<std::string>("push_function_", push_apply_func_decl->name);
                    //insert into MIR context
                    mir_context_->addFunctionFront(push_apply_func_decl);

                    node = hybrid_dense_edgeset_apply;
                  }
                }

                SimpleCPUScheduleObject::Ptr simple_schedule;
                if (!apply_schedule->isComposite()) {
                  simple_schedule = apply_schedule->self<SimpleCPUScheduleObject>();
                } else {
                  // arbitrarily grab first schedule object?
                  simple_schedule = apply_schedule->self<fir::abstract_schedule::CompositeScheduleObject>()->getFirstScheduleObject()->self<SimpleCPUScheduleObject>();
                }

                // No longer need this check as we moved the check to high-level scheduling API
                // We use negative integers between -1 and -10 to denote argv numbers
                // it can't be 0 as well, which indicates that this schedule is not needed
                if (simple_schedule->getNumSSG().getType() == fir::abstract_schedule::FlexIntVal::FlexIntType::ARG ||
                    simple_schedule->getNumSSG().getIntVal() != -100) {

                  std::string label_scope = label_scope_.getCurrentScope();
                  mir_context_->edgeset_to_label_to_num_segment[edgeset_expr->var.getName()][label_scope] =
                      simple_schedule->getNumSSG().getIntVal();
                  mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<std::string>("scope_label_name", label_scope);
                  mir_context_->edgeset_to_label_to_num_segment[edgeset_expr->var.getName()][label_scope] =
                      simple_schedule->getNumSSG().getIntVal();
                }

                //Check to see if it is parallel or serial
                if (simple_schedule->getCPUParallelizationType() != SimpleCPUScheduleObject::CPUParallelType::SERIAL) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("is_parallel", true);
                } else if (simple_schedule->getCPUParallelizationType() == SimpleCPUScheduleObject::CPUParallelType::SERIAL) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("is_parallel", false);
                }

                if (simple_schedule->getOutputQueueType() == SimpleCPUScheduleObject::OutputQueueType ::SLIDING_QUEUE) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_sliding_queue", true);
                } else {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_sliding_queue", false);
                }

                if (simple_schedule->getPullFrontierType() == SimpleScheduleObject::PullFrontierType::BITMAP) {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_pull_frontier_bitvector", true);
                } else {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_pull_frontier_bitvector", false);
                }

                if (simple_schedule->getParallelizationType() == SimpleScheduleObject::ParallelizationType ::EDGE_BASED){
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_pull_edge_based_load_balance", true);
                    if (simple_schedule->getPullLoadBalanceGrainSize().getIntVal() > 0){
                        mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<int>("pull_edge_based_load_balance_grain_size",
                                simple_schedule->getPullLoadBalanceGrainSize().getIntVal());
                    } else {
                        mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<int>("pull_edge_based_load_balance_grain_size", 4096);
                    }
                } else {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_pull_edge_based_load_balance", false);
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<int>("pull_edge_based_load_balance_grain_size", 4096);
                }

                //if this is applyModified with a tracking field
                if (edgeset_apply->tracking_field != "") {
                    // only enable deduplication when the argument to ApplyModified is True (disable deduplication), or the user manually set disable
                    if (edgeset_apply->getMetadata<bool>("enable_deduplication") && simple_schedule->getDeduplication() == SimpleScheduleObject::Deduplication ::ENABLED) {
                        //only enable deduplication if there is needed for tracking
                        mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("enable_deduplication", true);
                    }
                } else {
                    mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("enable_deduplication", false);
                }
            } else {
                //There is a schedule, but nothing is specified for the current apply
                node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
                mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_sliding_queue", false);
                mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_pull_frontier_bitvector", false);
                mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<bool>("use_pull_edge_based_load_balance", false);
                mir::to<mir::EdgeSetApplyExpr>(node)->setMetadata<int>("pull_edge_based_load_balance_grain_size", 4096);
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
