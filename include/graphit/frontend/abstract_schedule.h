//
// Created by chsue on 5/8/2020.
//

#ifndef GRAPHIT_ABSTRACT_SCHEDULE_H
#define GRAPHIT_ABSTRACT_SCHEDULE_H

#include <string>
#include <memory>
#include <assert.h>
#include "high_level_schedule.h"

namespace graphit {
    namespace fir {
        namespace abstract_schedule {
            class FlexIntVal {
            private:
                int val;
            public:
                FlexIntVal(int intVal = 0) {
                    val = intVal;
                    argv_idx = 0;
                }

                int getIntVal() {
                    return val;
                }

                FlexIntVal(std::string argv){
                  int argv_num = high_level_schedule::ProgramScheduleNode::extractArgvNumFromStringArg(argv);
                  val = argv_num;
              }
            };

        class ScheduleObject: public std::enable_shared_from_this<ScheduleObject>{
            public:
                typedef std::shared_ptr<ScheduleObject> Ptr;

                virtual bool isComposite() {};

                template<typename T>
                inline bool isa() {
                    return (bool) std::dynamic_pointer_cast<T>(shared_from_this());
                }

                template<typename T>
                inline const std::shared_ptr<T> to() {
                    std::shared_ptr<T> ret = std::dynamic_pointer_cast<T>(shared_from_this());
                    assert(ret != nullptr);
                    return ret;
                }

                template<typename T = ScheduleObject>
                std::shared_ptr<T> self() {
                    return to<T>();
                }
            };

            class SimpleScheduleObject : public ScheduleObject {
                // Abstract class has no functions for now
            public:
                typedef std::shared_ptr<SimpleScheduleObject> Ptr;
                enum class BackendID {
                    CPU,
                    GPU
                };

                enum class Direction {
                    PUSH,
                    PULL
                };

                enum class Deduplication {
                    ENABLED,
                    DISABLED
                };

                enum class ParallelizationType {
                    VERTEX_BASED,
                    EDGE_BASED
                };

                enum class PullFrontierType {
                    BITMAP,
                    BOOLMAP
                };

                virtual ParallelizationType getParallelizationType() {};

                virtual Direction getDirection() {};

                virtual Deduplication getDeduplication() {};

                virtual FlexIntVal getDelta() {};

                virtual PullFrontierType getPullFrontierType() {};

                virtual bool isComposite() override {
                    return false;
                }
            };

            class CompositeScheduleObject : public ScheduleObject {
            public:
                typedef std::shared_ptr<CompositeScheduleObject> Ptr;

                virtual ScheduleObject::Ptr getFirstScheduleObject() {};

                virtual ScheduleObject::Ptr getSecondScheduleObject() {};

                virtual bool isComposite() override {
                    return true;
                }
            };
        }
    }
}

#endif //GRAPHIT_ABSTRACT_SCHEDULE_H
