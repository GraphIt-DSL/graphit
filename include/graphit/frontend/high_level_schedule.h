//
// Created by Yunming Zhang on 6/13/17.
//

#ifndef GRAPHIT_HIGH_LEVEL_SCHEDULE_H
#define GRAPHIT_HIGH_LEVEL_SCHEDULE_H


#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include "fir.h"
#include <graphit/frontend/low_level_schedule.h>
#include <graphit/frontend/schedule.h>

namespace graphit {
    namespace fir {
        namespace high_level_schedule {

            using namespace std;

            class ProgramScheduleNode
                    : public std::enable_shared_from_this<ProgramScheduleNode> {

            public:

                ProgramScheduleNode(graphit::FIRContext *fir_context)
                        : fir_context_(fir_context) {
                    schedule_ = nullptr;
                }

                ~ ProgramScheduleNode(){
                    if (schedule_ != nullptr)
                        delete(schedule_);
                }

                typedef std::shared_ptr<ProgramScheduleNode> Ptr;


                // High level API for fusing together two fields / system vectors as ArrayOfStructs
                ProgramScheduleNode::Ptr fuseFields(string first_field_name,
                                                    string second_field_name);

                // High level API for splitting one loop into two loops
                ProgramScheduleNode::Ptr splitForLoop(string original_loop_label,
                                                      string split_loop1_label,
                                                      string split_loop2_label,
                                                      int split_loop1_range,
                                                      int split_loop2_range);

                //TODO: add high level APIs for fuseForLoops and fuseApplyFunctions
                //The APIs are documented here https://docs.google.com/document/d/1y-W8HkQKs3pZr5JX5FiI3pIYt8TPbNIX41V0ThZeVTY/edit?usp=sharing
                //See test/c++/low_level_schedule_test.cpp for examples of implementing these functionalities
                //using low level schedule APIs


                // High lvel API for speicifying scheduling options for apply
                // Scheduling Options include push, pull, hybrid, enable_deduplication, disable_deduplication, parallel, serial
                high_level_schedule::ProgramScheduleNode::Ptr
                setApply(std::string apply_label, std::string apply_schedule);

                // High lvel API for speicifying scheduling options for vertexset
                // Scheduling Options include sparse, dense
                high_level_schedule::ProgramScheduleNode::Ptr
                setVertexSet(std::string vertexset_label, std::string vertexset_schedule_str);

                Schedule * getSchedule() {
                    return  schedule_;
                }

            private:
                graphit::FIRContext * fir_context_;
                Schedule * schedule_;
            };


        }
    }
}

#endif //GRAPHIT_HIGH_LEVEL_SCHEDULE_H
