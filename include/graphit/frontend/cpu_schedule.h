//
// Created by chsue on 5/11/2020.
//

#ifndef GRAPHIT_CPU_SCHEDULE_H
#define GRAPHIT_CPU_SCHEDULE_H

#include <graphit/frontend/abstract_schedule.h>

namespace graphit {
    namespace fir {
        namespace cpu_schedule {
            class SimpleCPUScheduleObject : public abstract_schedule::SimpleScheduleObject {
            public:
                typedef std::shared_ptr<SimpleCPUScheduleObject> Ptr;
                enum class CPUPullFrontierType {
                    BITVECTOR,
                    BOOLMAP
                };

                enum class CPUParallelType {
                    WORK_STEALING_PAR,
                    STATIC_PAR,
                    SERIAL
                };

                enum class DirectionType {
                    PUSH,
                    PULL,
                    HYBRID_DENSE,
                    HYBRID_DENSE_FORWARD
                };

                enum class FrontierType {
                    SPARSE,
                    DENSE
                };

                enum class CPUDeduplicationType {
                    ENABLED,
                    DISABLED
                };

                enum class OutputQueueType {
                    QUEUE,
                    SLIDING_QUEUE
                };

                enum class CPUPullLoadBalanceType {
                    VERTEX_BASED,
                    EDGE_BASED
                };

                enum class PriorityUpdateType {
                    EAGER_PRIORITY_UPDATE,
                    EAGER_PRIORITY_UPDATE_WITH_MERGE,
                    CONST_SUM_REDUCTION_BEFORE_UPDATE,
                    REDUCTION_BEFORE_UPDATE
                };
            private:
                CPUPullLoadBalanceType cpu_pull_load_balance_type;
                CPUParallelType cpu_parallel_type;
                DirectionType direction_type;
                CPUDeduplicationType deduplication_type;
                OutputQueueType queue_type;
                CPUPullFrontierType cpu_pull_frontier_type;
                PriorityUpdateType priority_update_type;
                abstract_schedule::FlexIntVal pull_load_balance_grain_size;
                abstract_schedule::FlexIntVal num_segment;
                abstract_schedule::FlexIntVal delta;
                bool numa_aware;
                abstract_schedule::FlexIntVal merge_threshold;
                abstract_schedule::FlexIntVal num_open_buckets;
            public:
                SimpleCPUScheduleObject() {
                    cpu_pull_load_balance_type = CPUPullLoadBalanceType::VERTEX_BASED;
                    cpu_parallel_type = CPUParallelType ::SERIAL;
                    cpu_pull_frontier_type = CPUPullFrontierType :: BOOLMAP;
                    direction_type = DirectionType ::PUSH;
                    queue_type = OutputQueueType :: QUEUE;
                    deduplication_type = CPUDeduplicationType ::ENABLED;
                    priority_update_type = PriorityUpdateType ::REDUCTION_BEFORE_UPDATE;
                    pull_load_balance_grain_size = abstract_schedule::FlexIntVal(0);
                    num_segment = abstract_schedule::FlexIntVal(-100);
                    delta = abstract_schedule::FlexIntVal(1);
                    numa_aware = false;
                    merge_threshold = abstract_schedule::FlexIntVal(1000);
                    num_open_buckets = abstract_schedule::FlexIntVal(128);
                }

                virtual SimpleScheduleObject::ParallelizationType getParallelizationType() override {
                    if (cpu_pull_load_balance_type == CPUPullLoadBalanceType::VERTEX_BASED) {
                        return SimpleScheduleObject::ParallelizationType::VERTEX_BASED;
                    } else {
                        return SimpleScheduleObject::ParallelizationType ::EDGE_BASED;
                    }
                }

                virtual SimpleScheduleObject::Direction getDirection() override {
                    if (direction_type == DirectionType::PUSH || direction_type == DirectionType::HYBRID_DENSE_FORWARD) {
                        return SimpleScheduleObject::Direction ::PUSH;
                    } else if (direction_type == DirectionType::PULL){
                        return SimpleScheduleObject::Direction ::PULL;
                    } else {
                        // TODO(clhsu): Figure out what hybrid dense does
                    }
                }

                virtual SimpleScheduleObject::Deduplication getDeduplication() override {
                    if (deduplication_type == CPUDeduplicationType::ENABLED) {
                        return SimpleScheduleObject::Deduplication ::ENABLED;
                    } else {
                        return SimpleScheduleObject::Deduplication ::DISABLED;
                    }
                }

                virtual abstract_schedule::FlexIntVal getDelta() override {
                    return delta;
                }

                virtual SimpleScheduleObject::PullFrontierType getPullFrontierType() override {
                    if (cpu_pull_frontier_type == CPUPullFrontierType::BOOLMAP){
                        return SimpleScheduleObject::PullFrontierType ::BOOLMAP;
                    } else {
                        return SimpleScheduleObject::PullFrontierType ::BITMAP;
                    }
                }

                abstract_schedule::FlexIntVal getPullLoadBalanceGrainSize() {
                    return pull_load_balance_grain_size;
                }

                abstract_schedule::FlexIntVal getMergeThreshold() {
                    return merge_threshold;
                }

                abstract_schedule::FlexIntVal getNumOpenBuckets() {
                    return num_open_buckets;
                }

                bool getNumaAware() {
                    return numa_aware;
                }

                DirectionType getCPUDirection() {
                    return direction_type;
                }

              void configCPUDirection(DirectionType direction) {
                  direction_type = direction;
              }

              void configDeduplication(bool dedup) {
                  if (dedup) {
                    deduplication_type = CPUDeduplicationType :: ENABLED;
                  } else {
                    deduplication_type = CPUDeduplicationType::DISABLED;
                  }
                }

              void configBucketMergeThreshold(int threshold) {
                  merge_threshold = abstract_schedule::FlexIntVal(threshold);
                }

              void configApplyPriorityUpdateDelta(int update_delta) {
                delta = abstract_schedule::FlexIntVal(update_delta);
              }

              void configApplyNumSSG(int ssg) {
                num_segment = abstract_schedule::FlexIntVal(ssg);
              }

              void configApplyParallelization(std::string apply_parallel) {
                if (apply_parallel == "dynamic-vertex-parallel") {
                  cpu_parallel_type = CPUParallelType ::WORK_STEALING_PAR;
                } else if (apply_parallel == "static-vertex-parallel") {
                  cpu_parallel_type = CPUParallelType ::STATIC_PAR;
                } else if (apply_parallel == "serial") {
                  cpu_parallel_type = CPUParallelType ::SERIAL;
                } else if (apply_parallel == "edge-aware-dynamic-vertex-parallel") {
                  cpu_parallel_type = CPUParallelType ::WORK_STEALING_PAR;
                }
              }
              void configQueueType(std::string queue) {
                  if (queue == "sliding_queue") {
                    queue_type = OutputQueueType::SLIDING_QUEUE;
                  } else if (queue == "queue") {
                    queue_type = OutputQueueType ::QUEUE;
                  }
                }

                CPUParallelType getCPUParallelizationType() {
                    return cpu_parallel_type;
                }

                PriorityUpdateType getPriorityUpdateType() {
                    return priority_update_type;
                }

                OutputQueueType getOutputQueueType() {
                    return queue_type;
                }
            };

            class HybridCPUScheduleObject : public abstract_schedule::CompositeScheduleObject {

            private:
                ScheduleObject::Ptr first_schedule;
                ScheduleObject::Ptr second_schedule;
            public:
                typedef std::shared_ptr<HybridCPUScheduleObject> Ptr;
                HybridCPUScheduleObject (ScheduleObject::Ptr first, ScheduleObject::Ptr second) {
                    first_schedule = first;
                    second_schedule = second;
                }
                virtual ScheduleObject::Ptr getFirstScheduleObject() override {
                    return first_schedule;
                }
                virtual ScheduleObject::Ptr getSecondScheduleObject() override {
                    return second_schedule;
                }

                void configApplyNumSSG(int ssg, std::string direction) {
                  if (direction == "all") {
                    first_schedule->to<SimpleCPUScheduleObject>()->configApplyNumSSG(ssg);
                    second_schedule->to<SimpleCPUScheduleObject>()->configApplyNumSSG(ssg);
                  }
                  if (first_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PUSH) {
                    if (direction == "push") {
                      first_schedule->to<SimpleCPUScheduleObject>()->configApplyNumSSG(ssg);
                    }
                  } else if (first_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PULL) {
                    if (direction == "pull") {
                      first_schedule->to<SimpleCPUScheduleObject>()->configApplyNumSSG(ssg);
                    }
                  }

                  if (second_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PUSH) {
                    if (direction == "push") {
                      second_schedule->to<SimpleCPUScheduleObject>()->configApplyNumSSG(ssg);
                    }
                  } else if (second_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PULL) {
                    if (direction == "pull") {
                      second_schedule->to<SimpleCPUScheduleObject>()->configApplyNumSSG(ssg);
                    }
                  }
                }
                void configApplyParallelization(std::string apply_parallel, std::string direction) {
                  if (direction == "all") {
                    first_schedule->to<SimpleCPUScheduleObject>()->configApplyParallelization(direction);
                    second_schedule->to<SimpleCPUScheduleObject>()->configApplyParallelization(direction);
                  }
                  if (first_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PUSH) {
                    if (direction == "push") {
                      first_schedule->to<SimpleCPUScheduleObject>()->configApplyParallelization(direction);
                    }
                  } else if (first_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PULL) {
                    if (direction == "pull") {
                      first_schedule->to<SimpleCPUScheduleObject>()->configApplyParallelization(direction);
                    }
                  }

                  if (second_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PUSH) {
                    if (direction == "push") {
                      second_schedule->to<SimpleCPUScheduleObject>()->configApplyParallelization(direction);
                    }
                  } else if (second_schedule->to<SimpleCPUScheduleObject>()->getDirection() == SimpleCPUScheduleObject::DirectionType::PULL) {
                    if (direction == "pull") {
                      second_schedule->to<SimpleCPUScheduleObject>()->configApplyParallelization(direction);
                    }
                  }
                }

              void configDeduplication(bool enable) {
                second_schedule->to<SimpleCPUScheduleObject>()->configDeduplication(enable);
                first_schedule->to<SimpleCPUScheduleObject>()->configDeduplication(enable);
              }

              void configQueueType(std::string queue_type) {
                second_schedule->to<SimpleCPUScheduleObject>()->configQueueType(queue_type);
                first_schedule->to<SimpleCPUScheduleObject>()->configQueueType(queue_type);
              }

              void configApplyPriorityUpdateDelta(int update_delta) {
                second_schedule->to<SimpleCPUScheduleObject>()->configApplyPriorityUpdateDelta(update_delta);
                first_schedule->to<SimpleCPUScheduleObject>()->configApplyPriorityUpdateDelta(update_delta);
              }

              void configBucketMergeThreshold(int threshold) {
                second_schedule->to<SimpleCPUScheduleObject>()->configBucketMergeThreshold(threshold);
                first_schedule->to<SimpleCPUScheduleObject>()->configBucketMergeThreshold(threshold);
              }
            };
        }
    }
}
#endif //GRAPHIT_CPU_SCHEDULE_H
