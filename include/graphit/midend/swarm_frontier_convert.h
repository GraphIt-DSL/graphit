#ifndef GRAPHIT_INCLUDE_GRAPHIT_MIDEND_SWARM_FRONTIER_CONVERT_H_
#define GRAPHIT_INCLUDE_GRAPHIT_MIDEND_SWARM_FRONTIER_CONVERT_H_

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <unordered_set>
namespace graphit {
 struct SwarmFrontierConvert: public mir::MIRVisitor {
   using mir::MIRVisitor::visit;

  SwarmFrontierConvert (MIRContext* mir_context): mir_context_(mir_context) {
  }
  struct SwitchWhileCaseFinder: public mir::MIRVisitor {
     using mir::MIRVisitor::visit;
     bool can_switch = true;
     std::string frontier_name;

     virtual void visit(mir::EdgeSetApplyExpr::Ptr) override;
     virtual void visit(mir::VertexSetApplyExpr::Ptr) override;
     virtual void visit(mir::AssignStmt::Ptr) override;
     virtual void visit(mir::VarDecl::Ptr) override;

   };
  void analyze(void);
  protected:
    virtual void visit(mir::WhileStmt::Ptr);
  private:
    MIRContext *mir_context_ = nullptr;
};

}
#endif //GRAPHIT_INCLUDE_GRAPHIT_MIDEND_SWARM_FRONTIER_CONVERT_H_
