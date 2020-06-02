//
// Created by Claire Hsu on 5/11/2020.
//

#ifndef INCLUDE_GRAPHIT_FRONTEND_CPU_SCHEDULE_H_
#define INCLUDE_GRAPHIT_FRONTEND_CPU_SCHEDULE_H_

#include <graphit/frontend/abstract_schedule.h>
#include <memory>
#include <string>

namespace graphit {
namespace fir {
namespace cpu_schedule {
class SimpleCPUScheduleObject :
    public abstract_schedule::SimpleScheduleObject {
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
    SPARSE_PUSH,
    DENSE_PULL,
    DENSE_PUSH
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
  SimpleCPUScheduleObject(std::string direction="DensePush") {
    cpu_pull_load_balance_type = CPUPullLoadBalanceType::VERTEX_BASED;
    cpu_parallel_type = CPUParallelType::SERIAL;
    cpu_pull_frontier_type = CPUPullFrontierType::BOOLMAP;
    queue_type = OutputQueueType::QUEUE;
    deduplication_type = CPUDeduplicationType::ENABLED;
    priority_update_type = PriorityUpdateType::REDUCTION_BEFORE_UPDATE;
    pull_load_balance_grain_size = abstract_schedule::FlexIntVal(0);
    num_segment = abstract_schedule::FlexIntVal(-100);
    delta = abstract_schedule::FlexIntVal(1);
    numa_aware = false;
    merge_threshold = abstract_schedule::FlexIntVal(1000);
    num_open_buckets = abstract_schedule::FlexIntVal(128);

    direction_type = DirectionType::DENSE_PUSH;
    if (direction == "SparsePush") direction_type = DirectionType ::SPARSE_PUSH;
    if (direction == "DensePull") direction_type = DirectionType ::DENSE_PULL;
  }

  static SimpleCPUScheduleObject::DirectionType translateDirection(std::string direction) {
    if (direction == "SparsePush") return SimpleCPUScheduleObject::DirectionType::SPARSE_PUSH;
    if (direction == "DensePush") return SimpleCPUScheduleObject::DirectionType::DENSE_PUSH;
    if (direction == "DensePull") return SimpleCPUScheduleObject::DirectionType::DENSE_PULL;
    assert(false && "Direction not defined.");
  }

  SimpleCPUScheduleObject::Ptr cloneSchedule() {
    SimpleCPUScheduleObject::Ptr new_object = std::make_shared<SimpleCPUScheduleObject>(*this);
    return new_object;
  }

  SimpleScheduleObject::ParallelizationType getParallelizationType() override {
    if (cpu_pull_load_balance_type == CPUPullLoadBalanceType::VERTEX_BASED) {
      return SimpleScheduleObject::ParallelizationType::VERTEX_BASED;
    } else {
      return SimpleScheduleObject::ParallelizationType::EDGE_BASED;
    }
  }

  SimpleScheduleObject::Direction getDirection() override {
    if (direction_type == DirectionType::PUSH
        || direction_type == DirectionType::DENSE_PUSH
        || direction_type == DirectionType::SPARSE_PUSH) {
      return SimpleScheduleObject::Direction::PUSH;
    } else if (direction_type == DirectionType::PULL
        || direction_type == DirectionType::DENSE_PULL) {
      return SimpleScheduleObject::Direction::PULL;
    } else {
      assert(false && "Direction does not map to push or pull.");
    }
  }

  SimpleScheduleObject::Deduplication getDeduplication() override {
    if (deduplication_type == CPUDeduplicationType::ENABLED) {
      return SimpleScheduleObject::Deduplication::ENABLED;
    } else {
      return SimpleScheduleObject::Deduplication::DISABLED;
    }
  }

  abstract_schedule::FlexIntVal getDelta() override {
    return delta;
  }

  SimpleScheduleObject::PullFrontierType getPullFrontierType() override {
    if (cpu_pull_frontier_type == CPUPullFrontierType::BOOLMAP) {
      return SimpleScheduleObject::PullFrontierType::BOOLMAP;
    } else {
      return SimpleScheduleObject::PullFrontierType::BITMAP;
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
      deduplication_type = CPUDeduplicationType::ENABLED;
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

  void configApplyNUMA(bool aware) {
    numa_aware = aware;
  }

  void configApplyParallelization(std::string apply_parallel) {
    if (apply_parallel == "dynamic-vertex-parallel") {
      cpu_parallel_type = CPUParallelType::WORK_STEALING_PAR;
    } else if (apply_parallel == "static-vertex-parallel") {
      cpu_parallel_type = CPUParallelType::STATIC_PAR;
    } else if (apply_parallel == "serial") {
      cpu_parallel_type = CPUParallelType::SERIAL;
    } else if (apply_parallel == "edge-aware-dynamic-vertex-parallel") {
      cpu_parallel_type = CPUParallelType::WORK_STEALING_PAR;
      cpu_pull_load_balance_type = CPUPullLoadBalanceType::EDGE_BASED;
    }
  }
  void configQueueType(std::string queue) {
    if (queue == "sliding_queue") {
      queue_type = OutputQueueType::SLIDING_QUEUE;
    } else if (queue == "queue") {
      queue_type = OutputQueueType::QUEUE;
    }
  }

  void configApplyPriorityUpdate(std::string priority_update) {
    if (priority_update == "lazy_priority_update") {
      priority_update_type = PriorityUpdateType ::REDUCTION_BEFORE_UPDATE;
    } else if (priority_update == "eager_priority_update") {
      priority_update_type = PriorityUpdateType ::EAGER_PRIORITY_UPDATE;
    } else if (priority_update == "eager_priority_update_with_merge") {
      priority_update_type = PriorityUpdateType ::EAGER_PRIORITY_UPDATE_WITH_MERGE;
    } else if (priority_update == "constant_sum_reduce_before_update") {
      priority_update_type = PriorityUpdateType ::CONST_SUM_REDUCTION_BEFORE_UPDATE;
    } else {
      assert(false && "Priority update type not recognized.");
    }
  }

  void configApplyDenseVertexSet(std::string config) {
    if (config == "bitvector") {
      cpu_pull_frontier_type = CPUPullFrontierType ::BITVECTOR;
    }
  }

  void configPullLoadBalanceGrainSize(int grain_size) {
    pull_load_balance_grain_size = grain_size;
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

class HybridCPUScheduleObject :
    public abstract_schedule::CompositeScheduleObject {
 private:
  ScheduleObject::Ptr first_schedule;
  ScheduleObject::Ptr second_schedule;

 public:
  typedef std::shared_ptr<HybridCPUScheduleObject> Ptr;
  HybridCPUScheduleObject(ScheduleObject::Ptr first,
      ScheduleObject::Ptr second) {
    first_schedule = first;
    second_schedule = second;
  }
  ScheduleObject::Ptr getFirstScheduleObject() override {
    return first_schedule;
  }
  ScheduleObject::Ptr getSecondScheduleObject() override {
    return second_schedule;
  }
};

}  // namespace cpu_schedule
}  // namespace fir
}  // namespace graphit
#endif  // INCLUDE_GRAPHIT_FRONTEND_CPU_SCHEDULE_H_
