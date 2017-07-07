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
        high_level_schedule::ProgramScheduleNode::fuseFields(std::string first_field_name,
                                                             std::string second_field_name) {
            if (schedule_ == nullptr){
                schedule_ = new Schedule();
            }

            // If no apply schedule has been constructed, construct a new one
            if (schedule_->physical_data_layouts == nullptr){
                schedule_->physical_data_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
            }

            string fused_struct_name = "struct_" + first_field_name + "_" + second_field_name;
            
            FieldVectorPhysicalDataLayout vector_a_layout = {first_field_name, FieldVectorDataLayoutType::STRUCT, fused_struct_name};
            FieldVectorPhysicalDataLayout vector_b_layout = {second_field_name, FieldVectorDataLayoutType::STRUCT, fused_struct_name};

            (*schedule_->physical_data_layouts)[first_field_name] = vector_a_layout;
            (*schedule_->physical_data_layouts)[second_field_name] = vector_b_layout;

            return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::setApply(std::string apply_label, std::string apply_schedule_str) {

            // If no schedule has been constructed, construct a new one
            if (schedule_ == nullptr){
                schedule_ = new Schedule();
            }

            // If no apply schedule has been constructed, construct a new one
            if (schedule_->apply_schedules == nullptr){
                schedule_->apply_schedules = new std::map<std::string, ApplySchedule>();
            }

            // If no schedule has been specified for the current label, create a new one
            ApplySchedule apply_schedule;

            if (schedule_->apply_schedules->find(apply_label) == schedule_->apply_schedules->end()){
                //Default schedule pull, serial
                (*schedule_->apply_schedules)[apply_label]
                        = {apply_label, ApplySchedule::DirectionType::PULL, ApplySchedule::ParType::Serial};
            }


            if(apply_schedule_str == "push"){
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::PUSH;
            } else if (apply_schedule_str == "pull"){
                (*schedule_->apply_schedules)[apply_label].direction_type = ApplySchedule::DirectionType::PULL;
            } else{
                std::cout << "unrecognized schedule for apply: " << apply_schedule_str << std::endl;
            }


            return this->shared_from_this();
        }

        high_level_schedule::ProgramScheduleNode::Ptr
        high_level_schedule::ProgramScheduleNode::setVertexSet(std::string vertexset_label,
                                                               std::string vertexset_schedule_str) {
            // If no schedule has been constructed, construct a new one
            if (schedule_ == nullptr){
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
    }
}