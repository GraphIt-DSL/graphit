//
// Created by Claire Hsu on 5/8/2020.
//

#ifndef INCLUDE_GRAPHIT_FRONTEND_ABSTRACT_SCHEDULE_H_
#define INCLUDE_GRAPHIT_FRONTEND_ABSTRACT_SCHEDULE_H_

#include <assert.h>
#include <string>
#include <memory>

namespace graphit {
namespace fir {
namespace abstract_schedule {
class FlexIntVal {
 private:
  int val;
 public:
  FlexIntVal(int intVal = 0) {
    val = intVal;
  }

  int getIntVal() {
    return val;
  }

  FlexIntVal(const char *d) {
    if (sscanf(d, "argv[%i]", &val) != 1) {
      assert(false && "Invalid option for FlexIntVal str argument.");
    }
  }
};

class ScheduleObject : public std::enable_shared_from_this<ScheduleObject> {
 public:
  typedef std::shared_ptr<ScheduleObject> Ptr;

  virtual bool isComposite() {}

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

  virtual ParallelizationType getParallelizationType() {}

  virtual Direction getDirection() {}

  virtual Deduplication getDeduplication() {}

  virtual FlexIntVal getDelta() {}

  virtual PullFrontierType getPullFrontierType() {}

  bool isComposite() override {
    return false;
  }
};

class CompositeScheduleObject : public ScheduleObject {
 public:
  typedef std::shared_ptr<CompositeScheduleObject> Ptr;

  virtual ScheduleObject::Ptr getFirstScheduleObject() {}

  virtual ScheduleObject::Ptr getSecondScheduleObject() {}

  bool isComposite() override {
    return true;
  }
};
}  // namespace abstract_schedule
}  // namespace fir
}  // namespace graphit

#endif  // INCLUDE_GRAPHIT_FRONTEND_ABSTRACT_SCHEDULE_H_
