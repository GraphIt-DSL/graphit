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
                            l2_range_domain, l1_body_blk,  split_loop1_label, "i");
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
                                                              string fused_loop_label)
        {
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

                assert(l1_body_blk->emitFIRNode() != nullptr);
                assert(l2_body_blk->emitFIRNode() != nullptr);
                assert(fir::to<fir::IntLiteral>(l1_domain->lower)->val == fir::to<fir::IntLiteral>(l2_domain->lower)->val);
                assert(fir::to<fir::IntLiteral>(l1_domain->upper)->val == fir::to<fir::IntLiteral>(l2_domain->upper)->val);

                //create and set bounds for l3_loop (the fused loop)
                fir::low_level_schedule::RangeDomain::Ptr l3_range_domain =
                                    std::make_shared<fir::low_level_schedule::RangeDomain>
                                        (fir::to<fir::IntLiteral>(l1_domain->lower)->val,
                                         fir::to<fir::IntLiteral>(l1_domain->upper)->val);

                fir::low_level_schedule::ForStmtNode::Ptr l3_loop =
                                    std::make_shared<fir::low_level_schedule::ForStmtNode>(l3_range_domain,
                                                                                           l1_body_blk,
                                                                                           fused_loop_label,
                                                                                           "i");

                l3_loop->appendLoopBody(l2_body_blk);
                schedule_program_node->insertBefore(l3_loop, original_loop_label1);
                schedule_program_node->removeLabelNode(original_loop_label1);
                schedule_program_node->removeLabelNode(original_loop_label2);

                return this->shared_from_this();
        }
    }
}
