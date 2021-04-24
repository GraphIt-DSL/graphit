#ifndef SWARM_OPT_LOWER_H_
#define SWARM_OPT_LOWER_H_

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <unordered_set>

namespace graphit {
class SwarmOptLower {
  public:
    MIRContext *mir_context_;
    SwarmOptLower(MIRContext* mir_context): mir_context_(mir_context) {
  }
  void lower(void);

  struct CoarseningAttacher : public mir::MIRVisitor {
    MIRContext *mir_context_;
    CoarseningAttacher(MIRContext* mir_context): mir_context_(mir_context) {
    }
    using mir::MIRVisitor::visit;

    virtual void visit(mir::PushEdgeSetApplyExpr::Ptr) override;
  };

  struct HintAttacher : public mir::MIRVisitor {
    MIRContext *mir_context_;
    HintAttacher(MIRContext* mir_context): mir_context_(mir_context) {
    }
    using mir::MIRVisitor::visit;

    virtual void visit(mir::PushEdgeSetApplyExpr::Ptr) override;
  };
};

}
#endif //SWARM_OPT_LOWER_H_
