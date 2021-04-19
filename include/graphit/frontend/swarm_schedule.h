//
// Created by chsue on 1/6/2021.
//

#ifndef GRAPHIT_INCLUDE_GRAPHIT_FRONTEND_SWARM_SCHEDULE_H_
#define GRAPHIT_INCLUDE_GRAPHIT_FRONTEND_SWARM_SCHEDULE_H_
#include <assert.h>
#include <graphit/frontend/abstract_schedule.h>
#include <memory>

namespace graphit {
namespace fir {
namespace swarm_schedule {

enum swarm_schedule_options {
  PUSH,
  PULL,
  ENABLED,
  DISABLED,
  EDGE_BASED,
  VERTEX_BASED,
  BITMAP,
  BOOLMAP,
  PRIOQUEUE,
  BUCKETQUEUE
};

class SwarmSchedule {
  // Abstract class has no functions for now
 public:
  // Virtual destructor to make the class polymorphic
  virtual ~SwarmSchedule() = default;
};

class SimpleSwarmSchedule : public SwarmSchedule,
                          public abstract_schedule::SimpleScheduleObject {
 public:
  typedef std::shared_ptr<SimpleSwarmSchedule> Ptr;
  enum class pull_frontier_rep_type {
    BITMAP,
    BOOLMAP
  };
  enum class direction_type {
    DIR_PUSH,
    DIR_PULL
  };

  enum class deduplication_type {
    DEDUP_DISABLED,
    DEDUP_ENABLED
  };

  enum class load_balancing_type {
    VERTEX_BASED,
    EDGE_BASED
  };
  
  enum class QueueType {
    PRIOQUEUE,
    BUCKETQUEUE
  };

 public:
  direction_type direction;
  pull_frontier_rep_type pull_frontier_rep;
  deduplication_type deduplication;
  load_balancing_type load_balancing;
  QueueType queue_type;
  int32_t delta;
  abstract_schedule::FlexIntVal flex_delta;

  SimpleSwarmSchedule() {
    direction = direction_type::DIR_PUSH;
    pull_frontier_rep = pull_frontier_rep_type::BOOLMAP;
    deduplication = deduplication_type::DEDUP_DISABLED;
    load_balancing = load_balancing_type::VERTEX_BASED;
    delta = 1;
    queue_type = QueueType::PRIOQUEUE;
    flex_delta = abstract_schedule::FlexIntVal(1);
  }

 public:
  SimpleSwarmSchedule::Ptr cloneSchedule() {
    SimpleSwarmSchedule::Ptr new_object = std::make_shared<SimpleSwarmSchedule>(*this);
    return new_object;
  }

  SimpleScheduleObject::ParallelizationType getParallelizationType() override {
    switch (load_balancing) {
      case load_balancing_type::VERTEX_BASED:
        return SimpleScheduleObject::ParallelizationType ::VERTEX_BASED;
      case load_balancing_type::EDGE_BASED:
        return SimpleScheduleObject::ParallelizationType ::EDGE_BASED;
      default:assert(false && "Invalid option for Swarm Deduplication\n");
        break;
    }
  }

  SimpleScheduleObject::Direction getDirection() override {
    switch (direction) {
      case direction_type::DIR_PULL:
        return SimpleScheduleObject::Direction::PULL;
      case direction_type::DIR_PUSH:
        return SimpleScheduleObject::Direction::PUSH;
      default:assert(false && "Invalid option for Swarm Direction\n");
        break;
    }
  }

  SimpleScheduleObject::Deduplication getDeduplication() override {
    switch (deduplication) {
      case deduplication_type::DEDUP_ENABLED:
        return SimpleScheduleObject::Deduplication::ENABLED;
      case deduplication_type::DEDUP_DISABLED:
        return SimpleScheduleObject::Deduplication::DISABLED;
      default:assert(false && "Invalid option for Swarm Deduplication\n");
        break;
    }
  }
  
  abstract_schedule::FlexIntVal getDelta() override {
    return flex_delta;
  }

  SimpleScheduleObject::PullFrontierType getPullFrontierType() override {
    switch (pull_frontier_rep) {
      case pull_frontier_rep_type::BITMAP:
        return SimpleScheduleObject::PullFrontierType::BITMAP;
      case pull_frontier_rep_type::BOOLMAP:
        return SimpleScheduleObject::PullFrontierType::BOOLMAP;
      default:assert(false && "Invalid option for Swarm Pull Frontier Type\n");
        break;
    }
  }

  ScheduleObject::BackendID getBackendId() override {
    return ScheduleObject::BackendID ::SWARM;
  }

  void configDirection(enum swarm_schedule_options o,
                       enum swarm_schedule_options r = BOOLMAP) {
    switch (o) {
      case PUSH: direction = direction_type::DIR_PUSH;
        break;
      case PULL: direction = direction_type::DIR_PULL;
        switch (r) {
          case BITMAP: pull_frontier_rep = pull_frontier_rep_type::BITMAP;
            break;
          case BOOLMAP: pull_frontier_rep = pull_frontier_rep_type::BOOLMAP;
            break;
          default: assert(false &&
                "Invalid option for Pull Frontier representation\n");
            break;
        }
        break;
      default: assert(false && "Invalid option for configDirection");
        break;
    }
  }

  void configDeduplication(enum swarm_schedule_options o) {
    switch (o) {
      case ENABLED: deduplication = deduplication_type::DEDUP_ENABLED;
        break;
      case DISABLED: deduplication = deduplication_type::DEDUP_DISABLED;
        break;
      default: assert(false && "Invalid option for configDeduplication");
        break;
    }
  }

  void configLoadBalance(enum swarm_schedule_options o) {
    switch (o) {
      case VERTEX_BASED: load_balancing = load_balancing_type::VERTEX_BASED;
        break;
      case EDGE_BASED: load_balancing = load_balancing_type::EDGE_BASED;
        break;
      default: assert(false && "Invalid option for configLoadBalance");
        break;
    }
  }

  void configDelta(int32_t d) {
    if (d <= 0)
      assert(false && "Invalid option for configDelta");
    delta = d;
    flex_delta = abstract_schedule::FlexIntVal(d);
  }
  void configDelta(const char *d) {
    if (sscanf(d, "argv[%i]", &delta) != 1) {
      assert(false && "Invalid option for configDelta");
    }
    delta *= -1;
    flex_delta = abstract_schedule::FlexIntVal(d);
  }

  void configQueueType(enum swarm_schedule_options o) {
    switch (o) {
      case PRIOQUEUE: queue_type = QueueType::PRIOQUEUE;
	break;
      case BUCKETQUEUE: queue_type = QueueType::BUCKETQUEUE;
	break;
      default: assert(false && "Invalid option for configQueueType");
	break;
    }
  }
};

}  // namespace swarm_schedule
}  // namespace fir
}  // namespace graphit
#endif //GRAPHIT_INCLUDE_GRAPHIT_FRONTEND_SWARM_SCHEDULE_H_
