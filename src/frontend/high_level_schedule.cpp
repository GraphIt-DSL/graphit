//
// Created by Yunming Zhang on 6/13/17.
//

#include <graphit/frontend/high_level_schedule.h>

namespace graphit {
    namespace fir {

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::splitForLoop(std::string original_loop_label,
                                                               std::string split_loop1_label,
                                                               std::string split_loop2_label,
                                                               int split_loop1_range,
                                                               int split_loop2_range) {

            // use the low level scheduling API to make clone the body of "l1" loop
            fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
                    = std::make_shared<fir::low_level_schedule::ProgramNode>(fir_context_);
            fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
                    = schedule_program_node->cloneLabelLoopBody(original_loop_label);

            assert(l1_body_blk->emitFIRNode() != nullptr);

            //create and set bounds for l2_loop and l3_loop
            fir::low_level_schedule::RangeDomain::Ptr l2_range_domain
                    = std::make_shared<fir::low_level_schedule::RangeDomain>(0, split_loop1_range);
            fir::low_level_schedule::RangeDomain::Ptr l3_range_domain
                    = std::make_shared<fir::low_level_schedule::RangeDomain>(split_loop1_range,
                                                                             split_loop1_range + split_loop2_range);

            //create two new splitted loop node with labels split_loop1_label and split_loop2_label
            fir::low_level_schedule::ForStmtNode::Ptr l2_loop
                    = std::make_shared<fir::low_level_schedule::ForStmtNode>(
                            l2_range_domain, l1_body_blk, split_loop1_label, "i");
            fir::low_level_schedule::ForStmtNode::Ptr l3_loop
                    = std::make_shared<fir::low_level_schedule::ForStmtNode>(
                            l3_range_domain, l1_body_blk, split_loop2_label, "i");

            //insert split_loop1 and split_loop2 back into the program right before original_loop_label
            schedule_program_node->insertBefore(l2_loop, original_loop_label);
            schedule_program_node->insertBefore(l3_loop, original_loop_label);

            //remove original_loop
            schedule_program_node->removeLabelNode(original_loop_label);

            return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::fuseForLoop(string original_loop_label1,
                                                              string original_loop_label2,
                                                              string fused_loop_label) {
            // use the low level scheduling API to make clone the body of the "l1" and "l2" loops
            fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
                    = std::make_shared<fir::low_level_schedule::ProgramNode>(fir_context_);
            fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk
                    = schedule_program_node->cloneLabelLoopBody(original_loop_label1);
            fir::low_level_schedule::StmtBlockNode::Ptr l2_body_blk
                    = schedule_program_node->cloneLabelLoopBody(original_loop_label2);
            fir::low_level_schedule::ForStmtNode::Ptr l1_for
                    = schedule_program_node->cloneForStmtNode(original_loop_label1);
            fir::low_level_schedule::ForStmtNode::Ptr l2_for
                    = schedule_program_node->cloneForStmtNode(original_loop_label2);


            fir::RangeDomain::Ptr l1_domain = l1_for->for_domain_->emitFIRRangeDomain();
            fir::RangeDomain::Ptr l2_domain = l2_for->for_domain_->emitFIRRangeDomain();

            int l0 = fir::to<fir::IntLiteral>(l1_domain->lower)->val;
            int u0 = fir::to<fir::IntLiteral>(l1_domain->upper)->val;
            int l1 = fir::to<fir::IntLiteral>(l2_domain->lower)->val;
            int u1 = fir::to<fir::IntLiteral>(l2_domain->upper)->val;

            assert(l1_body_blk->emitFIRNode() != nullptr);
            assert(l2_body_blk->emitFIRNode() != nullptr);

            assert((l1 <= u0) && (l0 <= u1) && "Loops cannot be fused because they are completely separate.");
            assert((l0 >= 0) && (u0 >= 0) && (l1 >= 0) && (u1 >= 0) && "Loop bounds must be positive.");

            int prologue_size = std::max(l0, l1) - std::min(l0, l1);

            // Generate the prologue of the fused loops.
            if (prologue_size != 0) {
                //create and set bounds for the prologue loop
                fir::low_level_schedule::RangeDomain::Ptr l3_prologue_range_domain =
                        std::make_shared<fir::low_level_schedule::RangeDomain>
                                (std::min(l0, l1), std::max(l0, l1));

                std::string prologue_label = fused_loop_label + "_prologue";

                fir::low_level_schedule::ForStmtNode::Ptr l3_prologue_loop;

                if (l0 < l1) {

                    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk_copy
                            = schedule_program_node->cloneLabelLoopBody(original_loop_label1);
                    fir::low_level_schedule::NameNode::Ptr l1_name_node_copy
                            = std::make_shared<fir::low_level_schedule::NameNode>(l1_body_blk_copy,
                                                                                  original_loop_label1);
                    fir::low_level_schedule::StmtBlockNode::Ptr l1_name_node_stmt_blk_node_copy
                            = std::make_shared<fir::low_level_schedule::StmtBlockNode>();
                    l1_name_node_stmt_blk_node_copy->appendFirStmt(l1_name_node_copy->emitFIRNode());

                    l3_prologue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_prologue_range_domain,
                                                                                   l1_name_node_stmt_blk_node_copy,
                                                                                   prologue_label,
                                                                                   "i");
                } else {
                    fir::low_level_schedule::StmtBlockNode::Ptr l2_body_blk_copy
                            = schedule_program_node->cloneLabelLoopBody(original_loop_label2);
                    fir::low_level_schedule::NameNode::Ptr l2_name_node_copy
                            = std::make_shared<fir::low_level_schedule::NameNode>(l2_body_blk_copy,
                                                                                  original_loop_label2);
                    fir::low_level_schedule::StmtBlockNode::Ptr l2_name_node_stmt_blk_node_copy
                            = std::make_shared<fir::low_level_schedule::StmtBlockNode>();
                    l2_name_node_stmt_blk_node_copy->appendFirStmt(l2_name_node_copy->emitFIRNode());

                    l3_prologue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_prologue_range_domain,
                                                                                   l2_name_node_stmt_blk_node_copy,
                                                                                   prologue_label,
                                                                                   "i");
                }

                schedule_program_node->insertBefore(l3_prologue_loop, original_loop_label1);
            }

            //create and set bounds for l3_loop (the fused loop)
            fir::low_level_schedule::RangeDomain::Ptr l3_range_domain =
                    std::make_shared<fir::low_level_schedule::RangeDomain>
                            (max(l0, l1), min(u0, u1));

            // constructing a new stmtblocknode with namenode that contains l1_loop_body
            fir::low_level_schedule::NameNode::Ptr l1_name_node
                    = std::make_shared<fir::low_level_schedule::NameNode>(l1_body_blk, original_loop_label1);
            fir::low_level_schedule::StmtBlockNode::Ptr l1_name_node_stmt_blk_node_copy
                    = std::make_shared<fir::low_level_schedule::StmtBlockNode>();
            l1_name_node_stmt_blk_node_copy->appendFirStmt(l1_name_node->emitFIRNode());

            // constructing a new stmtblocknode with namenode that contains l2_loop_body
            fir::low_level_schedule::NameNode::Ptr l2_name_node
                    = std::make_shared<fir::low_level_schedule::NameNode>(l2_body_blk, original_loop_label2);
            fir::low_level_schedule::StmtBlockNode::Ptr l2_name_node_stmt_blk_node_copy
                    = std::make_shared<fir::low_level_schedule::StmtBlockNode>();
            l2_name_node_stmt_blk_node_copy->appendFirStmt(l2_name_node->emitFIRNode());

            fir::low_level_schedule::ForStmtNode::Ptr l3_loop =
                    std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_range_domain,
                                                                           l1_name_node_stmt_blk_node_copy,
                                                                           fused_loop_label,
                                                                           "i");

            // appending the stmtblocknode containing l2_name_node
            l3_loop->appendLoopBody(l2_name_node_stmt_blk_node_copy);
            schedule_program_node->insertBefore(l3_loop, original_loop_label1);


            int epilogue_size = std::max(u0, u1) - std::min(u0, u1);

            // Generate the epilogue loop.
            if (epilogue_size != 0) {
                //create and set bounds for the prologue loop
                fir::low_level_schedule::RangeDomain::Ptr l3_epilogue_range_domain =
                        std::make_shared<fir::low_level_schedule::RangeDomain>
                                (std::min(u0, u1), std::max(u0, u1));

                std::string epilogue_label = fused_loop_label + "_epilogue";

                fir::low_level_schedule::ForStmtNode::Ptr l3_epilogue_loop;

                if (u0 < u1) {
                    fir::low_level_schedule::StmtBlockNode::Ptr l2_body_blk_copy
                            = schedule_program_node->cloneLabelLoopBody(original_loop_label2);
                    fir::low_level_schedule::NameNode::Ptr l2_name_node_copy
                            = std::make_shared<fir::low_level_schedule::NameNode>(l2_body_blk_copy,
                                                                                  original_loop_label2);
                    fir::low_level_schedule::StmtBlockNode::Ptr l2_name_node_stmt_blk_node_copy
                            = std::make_shared<fir::low_level_schedule::StmtBlockNode>();
                    l2_name_node_stmt_blk_node_copy->appendFirStmt(l2_name_node_copy->emitFIRNode());

                    l3_epilogue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_epilogue_range_domain,
                                                                                   l2_name_node_stmt_blk_node_copy,
                                                                                   epilogue_label,
                                                                                   "i");
                } else {
                    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk_copy
                            = schedule_program_node->cloneLabelLoopBody(original_loop_label1);
                    fir::low_level_schedule::NameNode::Ptr l1_name_node_copy
                            = std::make_shared<fir::low_level_schedule::NameNode>(l1_body_blk_copy,
                                                                                  original_loop_label1);
                    fir::low_level_schedule::StmtBlockNode::Ptr l1_name_node_stmt_blk_node_copy
                            = std::make_shared<fir::low_level_schedule::StmtBlockNode>();
                    l1_name_node_stmt_blk_node_copy->appendFirStmt(l1_name_node_copy->emitFIRNode());


                    l3_epilogue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_epilogue_range_domain,
                                                                                   l1_name_node_stmt_blk_node_copy,
                                                                                   epilogue_label,
                                                                                   "i");
                }

                schedule_program_node->insertBefore(l3_epilogue_loop, original_loop_label1);
            }

            schedule_program_node->removeLabelNode(original_loop_label1);
            schedule_program_node->removeLabelNode(original_loop_label2);

            return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::fuseFields(std::string first_field_name,
                                                             std::string second_field_name) {
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }

            // If no apply schedule has been constructed, construct a new one
            if (schedule_->physical_data_layouts == nullptr) {
                schedule_->physical_data_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
            }

            string fused_struct_name = "struct_" + first_field_name + "_" + second_field_name;

            FieldVectorPhysicalDataLayout vector_a_layout = {first_field_name, FieldVectorDataLayoutType::STRUCT,
                                                             fused_struct_name};
            FieldVectorPhysicalDataLayout vector_b_layout = {second_field_name, FieldVectorDataLayoutType::STRUCT,
                                                             fused_struct_name};

            (*schedule_->physical_data_layouts)[first_field_name] = vector_a_layout;
            (*schedule_->physical_data_layouts)[second_field_name] = vector_b_layout;

            return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::fuseFields(std::vector<std::string> fields) {
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }

            // If no apply schedule has been constructed, construct a new one
            if (schedule_->physical_data_layouts == nullptr) {
                schedule_->physical_data_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
            }

            string fused_struct_name = "struct_";
            bool first = true;
            for (std::string field_name : fields) {
                if (first) {
                    first = false;
                } else {
                    fused_struct_name += "_";

                }
                fused_struct_name += field_name;
            }
            for (std::string field_name : fields) {
                FieldVectorPhysicalDataLayout vector_layout = {field_name, FieldVectorDataLayoutType::STRUCT,
                                                               fused_struct_name};
                (*schedule_->physical_data_layouts)[field_name] = vector_layout;
            }

            return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::setApply(std::string apply_label,
                                                           std::string apply_schedule_str,
                                                           int parameter) {
            // If no schedule has been constructed, construct a new one
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }

            // If no apply schedule has been constructed, construct a new one
            if (schedule_->apply_schedules == nullptr) {
                schedule_->apply_schedules = new std::map<std::string, ApplySchedule>();
            }

            // If no schedule has been specified for the current label, create a new one
            if (schedule_->apply_schedules->find(apply_label) == schedule_->apply_schedules->end()) {
                //Default schedule pull, serial, -100 for number of segments (we use -1 to -10 for argv)
                (*schedule_->apply_schedules)[apply_label]
                        = createDefaultSchedule(apply_label);
            }

            if (apply_schedule_str == "pull_edge_based_load_balance") {
                (*schedule_->apply_schedules)[apply_label].pull_load_balance_type
                        = ApplySchedule::PullLoadBalance::EDGE_BASED;
                (*schedule_->apply_schedules)[apply_label].pull_load_balance_edge_grain_size = parameter;
            } else if (apply_schedule_str == "pull") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::PULL;
            } else if (apply_schedule_str == "hybrid_dense") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::HYBRID_DENSE;
            } else if (apply_schedule_str == "num_segment") {
                (*schedule_->apply_schedules)[apply_label].num_segment = parameter;
            } else if (apply_schedule_str == "delta") {
                (*schedule_->apply_schedules)[apply_label].delta = parameter;
            } else if (apply_schedule_str == "bucket_merge_threshold"){
                (*schedule_->apply_schedules)[apply_label].merge_threshold = parameter;
            } else if (apply_schedule_str == "num_open_buckets"){
                (*schedule_->apply_schedules)[apply_label].num_open_buckets = parameter;
            }  else if (apply_schedule_str == "grain_size"){
                (*schedule_->apply_schedules)[apply_label].grain_size = parameter;

            } else {
                std::cout << "unrecognized schedule for apply: " << apply_schedule_str << std::endl;
                exit(0);
            }

            return this->shared_from_this();

        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::setApply(std::string apply_label, std::string apply_schedule_str) {

            // If no schedule has been constructed, construct a new one
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }

            // If no apply schedule has been constructed, construct a new one
            if (schedule_->apply_schedules == nullptr) {
                schedule_->apply_schedules = new std::map<std::string, ApplySchedule>();
            }

            // If no schedule has been specified for the current label, create a new one

            if (schedule_->apply_schedules->find(apply_label) == schedule_->apply_schedules->end()) {
                //Default schedule pull, serial, -100 for number of segments (we use -1 to -10 for argv)
                (*schedule_->apply_schedules)[apply_label]
                        = createDefaultSchedule(apply_label);
            }


            if (apply_schedule_str == "push") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::PUSH;
            } else if (apply_schedule_str == "pull") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::PULL;
            } else if (apply_schedule_str == "hybrid_dense_forward") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::HYBRID_DENSE_FORWARD;
            } else if (apply_schedule_str == "hybrid_dense") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::HYBRID_DENSE;
            } else if (apply_schedule_str == "serial") {
                (*schedule_->apply_schedules)[apply_label].parallel_type = ApplySchedule::ParType::Serial;
            } else if (apply_schedule_str == "parallel") {
                (*schedule_->apply_schedules)[apply_label].parallel_type = ApplySchedule::ParType::Parallel;
            } else if (apply_schedule_str == "enable_deduplication") {
                (*schedule_->apply_schedules)[apply_label].deduplication_type = ApplySchedule::DeduplicationType::Enable;
            } else if (apply_schedule_str == "disable_deduplication") {
                (*schedule_->apply_schedules)[apply_label].deduplication_type = ApplySchedule::DeduplicationType::Disable;
            } else if (apply_schedule_str == "sliding_queue") {
                (*schedule_->apply_schedules)[apply_label].opt = ApplySchedule::OtherOpt::SLIDING_QUEUE;
            } else if (apply_schedule_str == "pull_frontier_bitvector") {
                (*schedule_->apply_schedules)[apply_label].pull_frontier_type = ApplySchedule::PullFrontierType::BITVECTOR;
            } else if (apply_schedule_str == "pull_edge_based_load_balance") {
                (*schedule_->apply_schedules)[apply_label].pull_load_balance_type
                        = ApplySchedule::PullLoadBalance::EDGE_BASED;
            } else if (apply_schedule_str == "numa_aware") {
                (*schedule_->apply_schedules)[apply_label].numa_aware = true;
            } else if (apply_schedule_str == "lazy_priority_update"){
                (*schedule_->apply_schedules)[apply_label].priority_update_type
                        = ApplySchedule::PriorityUpdateType::REDUCTION_BEFORE_UPDATE;
            } else if (apply_schedule_str == "eager_priority_update") {
                (*schedule_->apply_schedules)[apply_label].priority_update_type
                        = ApplySchedule::PriorityUpdateType::EAGER_PRIORITY_UPDATE;
            } else if (apply_schedule_str == "eager_priority_update_with_merge") {
                (*schedule_->apply_schedules)[apply_label].priority_update_type
                        = ApplySchedule::PriorityUpdateType::EAGER_PRIORITY_UPDATE_WITH_MERGE;
	    } else if (apply_schedule_str == "constant_sum_reduce_before_update") {
	        (*schedule_->apply_schedules)[apply_label].priority_update_type
		        = ApplySchedule::PriorityUpdateType::CONST_SUM_REDUCTION_BEFORE_UPDATE;
            } else {
                std::cout << "unrecognized schedule for apply: " << apply_schedule_str << std::endl;
                exit(0);
            }

            return this->shared_from_this();
        }

        //DEPRECATED
        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::setVertexSet(std::string vertexset_label,
                                                               std::string vertexset_schedule_str) {
            // If no schedule has been constructed, construct a new one
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }


            if (vertexset_schedule_str == "sparse") {
                schedule_->vertexset_data_layout[vertexset_label]
                        = VertexsetPhysicalLayout{vertexset_label, VertexsetPhysicalLayout::DataLayout::SPARSE};
            } else if (vertexset_schedule_str == "dense") {
                schedule_->vertexset_data_layout[vertexset_label]
                        = VertexsetPhysicalLayout{vertexset_label, VertexsetPhysicalLayout::DataLayout::DENSE};
            } else {
                std::cout << "unrecognized schedule for vertexset: " << vertexset_schedule_str << std::endl;
            }

            return this->shared_from_this();
        }

        /**
         * Fuse two apply functions.
         * @param original_apply_label1: label of the first apply function.
         * @param original_apply_label2: label of the second apply function.
         * @param fused_apply_label: label of the fused apply function.
         * @param fused_apply_name: name of the fused apply function.
         */
        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::fuseApplyFunctions(string original_apply_label1,
                                                                     string original_apply_label2,
                                                                     string fused_apply_name) {
            //use the low level APIs to fuse the apply functions
            fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
                    = std::make_shared<fir::low_level_schedule::ProgramNode>(fir_context_);

            // Get the nodes of the apply functions
            fir::low_level_schedule::ApplyNode::Ptr first_apply_node
                    = schedule_program_node->cloneApplyNode(original_apply_label1);
            fir::low_level_schedule::ApplyNode::Ptr second_apply_node
                    = schedule_program_node->cloneApplyNode(original_apply_label2);



            // create the fused function declaration.  The function declaration of the first
            // apply function will be used to create the declaration of the fused function,
            // so we will rename it then add the body of the second apply function to it and
            // finally we will replace its label
            fir::low_level_schedule::FuncDeclNode::Ptr first_apply_func_decl
                    = schedule_program_node->cloneFuncDecl(first_apply_node->getApplyFuncName());

            if (first_apply_func_decl == nullptr) {
                std::cout << "Error: unable to clone function: " << first_apply_node->getApplyFuncName() << std::endl;
                return nullptr;
            }

            fir::low_level_schedule::StmtBlockNode::Ptr second_apply_func_body
                    = schedule_program_node->cloneFuncBody(second_apply_node->getApplyFuncName());
            first_apply_func_decl->appendFuncDeclBody(second_apply_func_body);
            first_apply_func_decl->setFunctionName(fused_apply_name);

            // Update the code that calls the apply functions
            //first_apply_node->updateStmtLabel(fused_apply_label);
            first_apply_node->updateApplyFunc(fused_apply_name);


//            std::cout << "fir1: " << std::endl;
//            std::cout << *(fir_context_->getProgram());
//            std::cout << std::endl;

            // Insert the declaration of the fused function and a call to it
            schedule_program_node->insertAfter(first_apply_func_decl, second_apply_node->getApplyFuncName());

//            std::cout << "fir2: " << std::endl;
//            std::cout << *(fir_context_->getProgram());
//            std::cout << std::endl;

            //schedule_program_node->insertBefore(first_apply_node, original_apply_label2);
            // This is a change from the original design, here we actually replace the original label
            schedule_program_node->replaceLabel(first_apply_node, original_apply_label1);

//            std::cout << "fir3: " << std::endl;
//            std::cout << *(fir_context_->getProgram());
//            std::cout << std::endl;

//            // Remove the original label nodes
//            if (!schedule_program_node->removeLabelNode(original_apply_label1)) {
//                std::cout << "remove node: " << original_apply_label1 << " failed" << std::endl;
//            }

            if (!schedule_program_node->removeLabelNode(original_apply_label2)) {
                std::cout << "remove node: " << original_apply_label2 << " failed" << std::endl;
            }

            // for debuggin, print out the FIR after the modifications
//            std::cout << "fir4: " << std::endl;
//            std::cout << *(fir_context_->getProgram());
//            std::cout << std::endl;

            return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyDirection(std::string apply_label,
                                                                       std::string apply_direction) {
            // If no schedule has been constructed, construct a new one
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }


            // If no schedule has been specified for the current label, create a new one

            //Default schedule pull, serial
            auto gis_vec = new std::vector<GraphIterationSpace>();
            if (apply_direction == "SparsePush-DensePull" || apply_direction == "DensePull-SparsePush") {

                //configure the first SparsePush graph iteration space
                auto gis_first = GraphIterationSpace();
                gis_first.direction = GraphIterationSpace::Direction::Push;
                gis_first.setFTTag(GraphIterationSpace::Dimension::OuterIter, Tags::FT_Tag::SparseArray);
                gis_first.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BoolArray);
                gis_first.scheduling_api_direction = "SparsePush";


                //configure the second DensePull graph iteration space
                auto gis_sec = GraphIterationSpace();
                gis_sec.direction = GraphIterationSpace::Direction::Pull;
                gis_sec.setFTTag(GraphIterationSpace::Dimension::OuterIter, Tags::FT_Tag::BoolArray);
                gis_sec.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BoolArray);
                gis_sec.scheduling_api_direction = "DensePull";

                gis_vec->push_back(gis_first);
                gis_vec->push_back(gis_sec);


            } else if (apply_direction == "DensePush-SparsePush" || apply_direction == "SparsePush-DensePush") {
                //configure the first graph iteration space DensePush
                auto gis_first = GraphIterationSpace();
                gis_first.direction = GraphIterationSpace::Direction::Push;
                gis_first.setFTTag(GraphIterationSpace::Dimension::OuterIter, Tags::FT_Tag::BoolArray);
                gis_first.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BoolArray);
                gis_first.scheduling_api_direction = "DensePush";


                //configure the first graph iteration space SparsePush
                auto gis_sec = GraphIterationSpace();
                gis_sec.direction = GraphIterationSpace::Direction::Push;
                gis_sec.setFTTag(GraphIterationSpace::Dimension::OuterIter, Tags::FT_Tag::SparseArray);
                gis_sec.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BoolArray);
                gis_sec.scheduling_api_direction = "SparsePush";

                gis_vec->push_back(gis_first);
                gis_vec->push_back(gis_sec);

            } else if (apply_direction == "SparsePush") {

                auto gis = GraphIterationSpace();
                gis.direction = GraphIterationSpace::Direction::Push;
                gis.setFTTag(GraphIterationSpace::Dimension::OuterIter, Tags::FT_Tag::SparseArray);
                gis.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BoolArray);
                gis.scheduling_api_direction = "SparsePush";
                gis_vec->push_back(gis);


            } else if (apply_direction == "DensePull") {
                auto gis = GraphIterationSpace();
                gis.direction = GraphIterationSpace::Direction::Pull;
                gis.setFTTag(GraphIterationSpace::Dimension::OuterIter, Tags::FT_Tag::BoolArray);
                gis.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BoolArray);
                gis.scheduling_api_direction = "DensePull";
                gis_vec->push_back(gis);

            } else if (apply_direction == "DensePush") {

                auto gis = GraphIterationSpace();
                gis.direction = GraphIterationSpace::Direction::Push;
                gis.setFTTag(GraphIterationSpace::Dimension::OuterIter, Tags::FT_Tag::BoolArray);
                gis.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BoolArray);
                gis.scheduling_api_direction = "DensePush";
                gis_vec->push_back(gis);


            } else {
                std::cout << "unsupported direction: " << apply_direction << std::endl;
                throw "Unsupported Schedule!";

            }


            //direction overrides previous GIS
            // direction should be the first scheduling command
            (*schedule_->graph_iter_spaces)[apply_label] = gis_vec;


            //for now we still uses the old API, will slowly deprecated
            if (dirCompatibilityMap_.find(apply_direction) != dirCompatibilityMap_.end()) {
                std::string old_dir_schedule = dirCompatibilityMap_[apply_direction];
                return setApply(apply_label, old_dir_schedule);
            } else {
                return setApply(apply_label, apply_direction);
            }

        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configIntersection(std::string intersection_label,
                                                                       std::string intersection_option) {
            // If no schedule has been constructed, construct a new one
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }

            // If no intersection schedule has been constructed, construct a new one
            if (schedule_->intersection_schedules == nullptr) {
                schedule_->intersection_schedules = new std::map<std::string, IntersectionSchedule::IntersectionType>();
            }

            if (schedule_->intersection_schedules->find(intersection_label) == schedule_->intersection_schedules->end()) {
                (*schedule_->intersection_schedules)[intersection_label] = IntersectionSchedule::IntersectionType::NAIVE;
            }


            if (intersection_option == "HiroshiIntersection") {
                (*schedule_->intersection_schedules)[intersection_label] = IntersectionSchedule::IntersectionType::HIROSHI;
            }

            else if (intersection_option == "MultiskipIntersection") {
                (*schedule_->intersection_schedules)[intersection_label] = IntersectionSchedule::IntersectionType::MULTISKIP;
            }

            else if (intersection_option == "CombinedIntersection") {
                (*schedule_->intersection_schedules)[intersection_label] = IntersectionSchedule::IntersectionType::COMBINED;
            }

            else if (intersection_option == "BinarySearchIntersection") {
                (*schedule_->intersection_schedules)[intersection_label] = IntersectionSchedule::IntersectionType::BINARY;
            }
            else if (intersection_option == "NaiveIntersection") {
                (*schedule_->intersection_schedules)[intersection_label] = IntersectionSchedule::IntersectionType::NAIVE;
            }

            // if no valid intersection is specified, we use naive intersection.
            else {
                std::cout << "unsupported intersection: " << intersection_option << " using naive version instead" << std::endl;
                (*schedule_->intersection_schedules)[intersection_label] = IntersectionSchedule::IntersectionType::NAIVE;
            }

            return this->shared_from_this();


        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyParallelization(std::string apply_label,
                                                                             std::string apply_parallel, int grain_size,
                                                                             std::string direction) {


            initGraphIterationSpaceIfNeeded(apply_label);


            auto gis_vec = (*schedule_->graph_iter_spaces)[apply_label];

            for (auto &gis : *gis_vec) {
                if (gis.scheduling_api_direction == direction || direction == "all") {
                    //configure only the GIS identified by the directin. By default "all" configures all the directions
                    if (apply_parallel == "dynamic-vertex-parallel") {
                        gis.setPRTag(GraphIterationSpace::Dimension::BSG, Tags::PR_Tag::WorkStealingPar);
                    } else if (apply_parallel == "static-vertex-parallel") {
                        gis.setPRTag(GraphIterationSpace::Dimension::BSG, Tags::PR_Tag::StaticPar);
                    } else if (apply_parallel == "serial") {
                        gis.setPRTag(GraphIterationSpace::Dimension::BSG, Tags::PR_Tag::Serial);
                    } else if (apply_parallel == "edge-aware-dynamic-vertex-parallel") {
                        gis.setPTTag(GraphIterationSpace::Dimension::BSG, Tags::PT_Tag::EdgeAwareVertexCount);
                        gis.setPRTag(GraphIterationSpace::Dimension::BSG, Tags::PR_Tag::WorkStealingPar);
                    } else {
                        std::cout << "unsupported parallelization strategy: " << apply_parallel << std::endl;
                        throw "Unsupported Schedule!";

                    }

                    if (grain_size != 1024) {
                        //if the grain size is not the default size
                        gis.BSG_grain_size = grain_size;
                    }
                }
            }

            // set apply config
            setApply(apply_label, "grain_size", grain_size);
            //for now, we still use the old API, it will slowly be deprecated
            if (parallelCompatibilityMap_.find(apply_parallel) != parallelCompatibilityMap_.end()) {
                std::string old_par_schedule = parallelCompatibilityMap_[apply_parallel];
                if (apply_parallel == "edge-aware-dynamic-vertex-parallel") {
                    //need a separate specification in the old API
                    setApply(apply_label, "pull_edge_based_load_balance");
                    return setApply(apply_label, old_par_schedule);
                } else {
                    return setApply(apply_label, old_par_schedule);
                }
            } else {
                return setApply(apply_label, apply_parallel);
            }
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyDenseVertexSet(std::string label, std::string config,
                                                                            std::string vertexset,
                                                                            std::string direction) {

            initGraphIterationSpaceIfNeeded(label);
            auto gis_vec = (*schedule_->graph_iter_spaces)[label];

            //right now, we only support configuring the source vertexset in the pull direction
            for (auto &gis : *gis_vec) {
                if (gis.scheduling_api_direction == direction || direction == "all") {
                    if (gis.direction == GraphIterationSpace::Direction::Push) {
                        //TODO: we don't
                        if (vertexset == "src-vertexset") {
                            std::cout << "unsupported dense configuration for vertexset: " << vertexset
                                      << " direction: " << direction << std::endl;
                            throw "Unsupported Schedule!";
                        } else {
                            std::cout << "unsupported dense configuration for vertexset: " << vertexset
                                      << " direction: " << direction << std::endl;
                            throw "Unsupported Schedule!";
                        }
                    } else {
                        if (vertexset == "src-vertexset") {
                            if (config == "bitvector"){
                                gis.setFTTag(GraphIterationSpace::Dimension::InnerITer, Tags::FT_Tag::BitVector);
                            }
                        } else {
                            //TODO: we don't yet support configuring densePull dst-vertexset
                            std::cout << "unsupported dense configuration for vertexset: " << vertexset
                                      << " direction: " << direction << std::endl;
                            throw "Unsupported Schedule!";
                        }
                    }
                }
            }

            // to use the old API, will slowly be deprecated
            if (config == "bitvector") {
                return setApply(label, "pull_frontier_bitvector");
            } else {
                return this->shared_from_this();
            }

        }

        // extract the integer from a string
        int high_level_schedule::ProgramScheduleNode::extractIntegerFromString(string input_string){

            std::size_t const n = input_string.find_first_of("0123456789");
            if (n != std::string::npos)
            {
                std::size_t const m = input_string.find_first_not_of("0123456789", n);
                return stoi(input_string.substr(n, m != std::string::npos ? m-n : m));
            }
            return -1;

        }


        // use string rfind insted of regular expression because gcc older than 4.9.0 does not support regular expression
        //regex argv_regex ("argv\\[(\\d)\\]");

        // here we do a hack and uses a negative integer to denote the integer argument to argv
        // the code generation will treat negative numbers differently by generating a argv[negative_integer) run time argument
        // to use as number of segments
        // the user input argv string has to match a pattern argv[integer]
        //if (regex_match(num_segment_argv, argv_regex)){
        int high_level_schedule::ProgramScheduleNode::extractArgvNumFromStringArg(string argv_str) {
            int argv_number;
            if (argv_str.rfind("argv[", 0) == 0){
                argv_number = -1*extractIntegerFromString(argv_str);
            } else {
                std::cerr <<  "Invalid string argument. It has to be of form argv[integer]" << std::endl;
                throw "Unsupported Schedule!";
            }
            return argv_number;
        }


        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyPriorityUpdate(std::string apply_label, std::string config) {
            initGraphIterationSpaceIfNeeded(apply_label);

            // for now, we still use the old setApply API. We will probably switch to full graph iteration space soon
            return setApply(apply_label, config);
        }



        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyNumSSG(std::string apply_label, std::string config,
                                                                    string num_segment_argv, std::string direction) {

            initGraphIterationSpaceIfNeeded(apply_label);
            auto gis_vec = (*schedule_->graph_iter_spaces)[apply_label];
            int argv_number;

            for (auto &gis : *gis_vec) {
                if (gis.scheduling_api_direction == direction || direction == "all") {
                    if (config == "fixed-vertex-count"){
                        if (gis.scheduling_api_direction != "DensePull"){
                            //currently, we don't support any direction other than DensePull for graph partitioning
                            // push based partitioning is coming
                            std::cout << "unsupported direction for partition SSGs: "  << gis.scheduling_api_direction << std::endl;
                            throw "Unsupported Schedule!";
                        }
                        gis.setPTTag(GraphIterationSpace::Dimension::SSG, Tags::PT_Tag::FixedVertexCount);
                    } else if (config == "edge-aware-vertex-count"){
                        gis.setPTTag(GraphIterationSpace::Dimension::SSG, Tags::PT_Tag::EdgeAwareVertexCount);
                        throw "Unsupported Schedule!";
                    } else {
                        throw "Unsupported Schedule!";
                    }

                    argv_number = extractArgvNumFromStringArg(num_segment_argv);

                    //gis is not really used right now
                    gis.num_ssg = argv_number;
                }
            }

            // for now, we still use the old setApply API. We will probably switch to full graph iteration space soon
            return setApply(apply_label, "num_segment", argv_number);

        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyNumSSG(std::string apply_label, std::string config,
                                                                    int num_segment, std::string direction) {


            initGraphIterationSpaceIfNeeded(apply_label);
            auto gis_vec = (*schedule_->graph_iter_spaces)[apply_label];

            for (auto &gis : *gis_vec) {
                if (gis.scheduling_api_direction == direction || direction == "all") {
                    if (config == "fixed-vertex-count"){
                        if (gis.scheduling_api_direction != "DensePull"){
                            //currently, we don't support any direction other than DensePull for graph partitioning
                            // push based partitioning is coming
                            std::cout << "unsupported direction for partition SSGs: "  << gis.scheduling_api_direction << std::endl;
                            throw "Unsupported Schedule!";
                        }
                        gis.setPTTag(GraphIterationSpace::Dimension::SSG, Tags::PT_Tag::FixedVertexCount);
                    } else if (config == "edge-aware-vertex-count"){
                        gis.setPTTag(GraphIterationSpace::Dimension::SSG, Tags::PT_Tag::EdgeAwareVertexCount);
                        throw "Unsupported Schedule!";
                    } else {
                        throw "Unsupported Schedule!";
                    }
                    assert(num_segment > 0);
                    gis.num_ssg = num_segment;
                }
            }

            return setApply(apply_label, "num_segment", num_segment);
        }

        void high_level_schedule::ProgramScheduleNode::initGraphIterationSpaceIfNeeded(std::string label) {
            if (schedule_ == nullptr) {
                schedule_ = new Schedule();
            }

            if ((*schedule_->graph_iter_spaces).find(label) == (*schedule_->graph_iter_spaces).end()) {
                //if there's no graph iteration space associated with the label
                auto gis_vec = new std::vector<GraphIterationSpace>();
                (*schedule_->graph_iter_spaces)[label] = gis_vec;
                gis_vec->push_back(GraphIterationSpace());
            }
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyNUMA(std::string apply_label, std::string config,
                                                                  std::string direction) {

            initGraphIterationSpaceIfNeeded(apply_label);
            auto gis_vec = (*schedule_->graph_iter_spaces)[apply_label];
            for (auto &gis : *gis_vec) {
                if (gis.scheduling_api_direction == direction || direction == "all") {
                    if (gis.scheduling_api_direction != "DensePull"){
                        //currently, we don't support any direction other than DensePull for graph partitioning
                        // push based partitioning is coming
                        std::cout << "unsupported direction for NUMA optimizations: "  << gis.scheduling_api_direction << std::endl;
                        throw "Unsupported Schedule!";
                    }

                    if (config == "serial"){
                        gis.setPRTag(GraphIterationSpace::Dimension::SSG, Tags::PR_Tag::Serial);
                    } else if (config == "static-parallel"){
                        gis.setPRTag(GraphIterationSpace::Dimension::SSG, Tags::PR_Tag::StaticPar);
                    } else if (config == "dynamic-parallel"){
                        gis.setPRTag(GraphIterationSpace::Dimension::SSG, Tags::PR_Tag::WorkStealingPar);
                    } else {
                        throw "Unsupported Schedule!";
                    }
                }
            }
            // this is using the old API, will be deprecated soon
            if (config == "static-parallel")
                return setApply(apply_label, "numa_aware");
            else
                return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyPriorityUpdateDelta(std::string apply_label, int delta) {
            return setApply(apply_label, "delta", delta);
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configBucketMergeThreshold(std::string apply_label, int threshold) {
            return setApply(apply_label, "bucket_merge_threshold", threshold);
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configApplyPriorityUpdateDelta(std::string apply_label,
                                                                                 std::string delta_argv) {

            int argv_num = extractArgvNumFromStringArg(delta_argv);
            return setApply(apply_label, "delta", argv_num);
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configBucketMergeThreshold(std::string apply_label, string threshold_argv) {
            int argv_num = extractArgvNumFromStringArg(threshold_argv);
            return setApply(apply_label, "bucket_merge_threshold", argv_num);
        }

        // Create a default schedule parameters
        ApplySchedule high_level_schedule::ProgramScheduleNode::createDefaultSchedule(std::string apply_label) {
            return {apply_label, ApplySchedule::DirectionType::PUSH, // default direction is push
                    ApplySchedule::ParType::Serial,
                    ApplySchedule::DeduplicationType::Enable,
                    ApplySchedule::OtherOpt::QUEUE,
                    ApplySchedule::PullFrontierType::BOOL_MAP,
                    ApplySchedule::PullLoadBalance::VERTEX_BASED,
                    ApplySchedule::PriorityUpdateType::REDUCTION_BEFORE_UPDATE,
                    0, // pull_load_balance_edge_grain_size
                    -100, // num_segment
                    1, // default delta
                    256,
                    false, // enable_numa_aware?
                    1000, // merge threshold for eager prioirty queue
                    128   // default number of open buckets for lazy priority queue
            };
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configNumOpenBuckets(std::string apply_label, int num_open_buckets) {
            return setApply(apply_label, "num_open_buckets", num_open_buckets);
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::configNumOpenBuckets(std::string apply_label,
                                                                                 std::string num_open_buckets) {

            int argv_num = extractArgvNumFromStringArg(num_open_buckets);
            return setApply(apply_label, "num_open_buckets", argv_num);
        }

    }
}
