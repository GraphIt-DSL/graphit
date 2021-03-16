#ifndef SWARM_PRIORITY_FEATURES_LOWERING_H_
#define SWARM_PRIORITY_FEATURES_LOWERING_H_

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <unordered_set>
namespace graphit {

class SwarmPriorityFeaturesLowering {
  public:
	  MIRContext *mir_context_;
	  SwarmPriorityFeaturesLowering (MIRContext* mir_context):mir_context_(mir_context) {
	  }
	  void lower(void);
	  struct PrioFrontierFinderVisitor: public mir::MIRVisitor {
	    using mir::MIRVisitor::visit;
	    MIRContext *mir_context_;
	    virtual void visit(mir::WhileStmt::Ptr) override;
	  };
	  struct WhileStmtPQVisitor: public mir::MIRVisitor {
	    MIRContext *mir_context_;
	    
	    using mir::MIRVisitor::visit;
	    bool to_delete;
	    std::vector<mir::Stmt::Ptr> to_deletes;
	    std::string temp_frontier_name = "";
	    std::string pq_name;
	    virtual void visit(mir::StmtBlock::Ptr) override;
	    virtual void visit(mir::Call::Ptr) override;
	    virtual void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr) override;
	    virtual void visit(mir::VarDecl::Ptr) override;	    
	  };
};
}
#endif //SWARM_PRIORITY_FEATURES_LOWERING_H_
