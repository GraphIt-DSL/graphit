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

namespace graphit {
    namespace fir {
        namespace high_level_schedule {

            using namespace std;

            class ProgramScheduleNode
                    : public std::enable_shared_from_this<ProgramScheduleNode> {

            public:

                ProgramScheduleNode(graphit::FIRContext *fir_context)
                        : fir_context_(fir_context) {

                }

                typedef std::shared_ptr<ProgramScheduleNode> Ptr;

                ProgramScheduleNode::Ptr splitForLoop(string original_loop_label,
                                                      string split_loop1_label,
                                                      string split_loop2_label,
                                                      int split_loop1_range,
                                                      int split_loop2_range);

            private:
                graphit::FIRContext * fir_context_;
            };


        }
    }
}

#endif //GRAPHIT_HIGH_LEVEL_SCHEDULE_H
