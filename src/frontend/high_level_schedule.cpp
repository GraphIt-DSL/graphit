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

                    l3_prologue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_prologue_range_domain,
                                                                                   l1_body_blk_copy,
                                                                                   prologue_label,
                                                                                   "i");
                } else {
                    fir::low_level_schedule::StmtBlockNode::Ptr l2_body_blk_copy
                            = schedule_program_node->cloneLabelLoopBody(original_loop_label2);

                    l3_prologue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_prologue_range_domain,
                                                                                   l2_body_blk_copy,
                                                                                   prologue_label,
                                                                                   "i");
                }

                schedule_program_node->insertBefore(l3_prologue_loop, original_loop_label1);
            }

            //create and set bounds for l3_loop (the fused loop)
            fir::low_level_schedule::RangeDomain::Ptr l3_range_domain =
                    std::make_shared<fir::low_level_schedule::RangeDomain>
                            (max(l0, l1), min(u0, u1));

            fir::low_level_schedule::ForStmtNode::Ptr l3_loop =
                    std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_range_domain,
                                                                           l1_body_blk,
                                                                           fused_loop_label,
                                                                           "i");

            l3_loop->appendLoopBody(l2_body_blk);
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

                    l3_epilogue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_epilogue_range_domain,
                                                                                   l2_body_blk_copy,
                                                                                   epilogue_label,
                                                                                   "i");
                } else {
                    fir::low_level_schedule::StmtBlockNode::Ptr l1_body_blk_copy
                            = schedule_program_node->cloneLabelLoopBody(original_loop_label1);

                    l3_epilogue_loop =
                            std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_epilogue_range_domain,
                                                                                   l1_body_blk_copy,
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
            ApplySchedule apply_schedule;

            if (schedule_->apply_schedules->find(apply_label) == schedule_->apply_schedules->end()) {
                //Default schedule pull, serial
                (*schedule_->apply_schedules)[apply_label]
                        = {apply_label, ApplySchedule::DirectionType::PULL, ApplySchedule::ParType::Serial};
            }


            if (apply_schedule_str == "push") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::PUSH;
            } else if (apply_schedule_str == "pull") {
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::PULL;
            } else if (apply_schedule_str == "sparse_frontier") {
                (*schedule_->apply_schedules)[apply_label].frontier_type = ApplySchedule::FrontierType::Sparse;
            } else {
                std::cout << "unrecognized schedule for apply: " << apply_schedule_str << std::endl;
            }


            return this->shared_from_this();
        }

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

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::fuseApplyFunctions(string original_apply_label1,
                                                                     string original_apply_label2,
                                                                     string fused_apply_label,
                                                                     string fused_apply_name)
        {
            //use the low level APIs to fuse the apply functions
             fir::low_level_schedule::ProgramNode::Ptr schedule_program_node
                     = std::make_shared<fir::low_level_schedule::ProgramNode>(fir_context_);
             fir::low_level_schedule::ApplyNode::Ptr first_apply_node
                     = schedule_program_node->cloneApplyNode(original_apply_label1);
             fir::low_level_schedule::ApplyNode::Ptr second_apply_node
                     = schedule_program_node->cloneApplyNode(original_apply_label2);
             first_apply_node->updateStmtLabel(fused_apply_label);

             // create a fused function
             fir::low_level_schedule::FuncDeclNode::Ptr first_apply_func_decl
                     = schedule_program_node->cloneFuncDecl(first_apply_node->getApplyFuncName());
             fir::low_level_schedule::StmtBlockNode::Ptr second_apply_func_body
                     = schedule_program_node->cloneFuncBody(second_apply_node->getApplyFuncName());

             first_apply_func_decl->appendFuncDeclBody(second_apply_func_body);
             first_apply_func_decl->setFunctionName(fused_apply_name);

             schedule_program_node->insertAfter(first_apply_func_decl, second_apply_node->getApplyFuncName());

             first_apply_node->updateApplyFunc(fused_apply_name);

             schedule_program_node->insertBefore(first_apply_node, original_apply_label2);
             schedule_program_node->removeLabelNode(original_apply_label1);
             schedule_program_node->removeLabelNode(original_apply_label2);

             return this->shared_from_this();
        }
    }
}