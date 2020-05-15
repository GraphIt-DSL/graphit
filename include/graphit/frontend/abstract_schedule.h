//
// Created by chsue on 5/8/2020.
//

#ifndef GRAPHIT_ABSTRACT_SCHEDULE_H
#define GRAPHIT_ABSTRACT_SCHEDULE_H

#include <string>
#include <memory>
#include <assert.h>

namespace graphit {
    class FlexIntVal {
    private:
        int val;
        int argv_idx;
    public:
        FlexIntVal(int intVal = 0) {
            val = intVal;
            argv_idx = 0;
        }

        int getIntVal() {
            return val;
        }

//        FlexIntVal(std::string argv){
//            // TODO(clhsu): parse something here ??
//        }
    };

    namespace fir {
        namespace abstract_schedule {
            class ScheduleObject {
            public:
                typedef std::shared_ptr<ScheduleObject> Ptr;

                virtual bool isComposite() {};
            };

            template<typename T>
            inline bool isa(std::shared_ptr<ScheduleObject> ptr) {
                return (bool) std::dynamic_pointer_cast<T>(ptr);
            }

            template<typename T>
            inline const std::shared_ptr<T> to(std::shared_ptr<ScheduleObject> ptr) {
                std::shared_ptr<T> ret = std::dynamic_pointer_cast<T>(ptr);
                assert(ret != nullptr);
                return ret;
            }

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
