#ifndef FRONTIER_DEDUP_LOWER_H_
#define FRONTIER_DEDUP_LOWER_H_

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <unordered_set>
namespace graphit {
class FrontierDedupLower {
 public:
  MIRContext *mir_context_;
  FrontierDedupLower (MIRContext* mir_context): mir_context_(mir_context) {
  }
  void lower(void);
  struct ReuseFrontierFinderVisitor: public mir::MIRVisitor {
    MIRContext *mir_context_;
    ReuseFrontierFinderVisitor(MIRContext* mir_context): mir_context_(mir_context) {
    }
    using mir::MIRVisitor::visit;
    std::vector<mir::Stmt::Ptr> to_deletes;
    bool is_reflexive_expr(mir::AssignStmt::Ptr assign_stmt);
    virtual void visit(mir::StmtBlock::Ptr) override;
  };

  struct FrontierVarChangeVisitor: public mir::MIRVisitor {
    using mir::MIRVisitor::visit;
    MIRContext *mir_context_;
    std::string frontier_name;
    std::string old_frontier_name;

    virtual void visit(mir::AssignStmt::Ptr) override;
    virtual void visit(mir::VarDecl::Ptr) override;
    virtual void visit(mir::VertexSetApplyExpr::Ptr) override;
    virtual void visit(mir::Call::Ptr) override;
  };

  struct VertexDeduplicationVisitor: public mir::MIRVisitor {
    using mir::MIRVisitor::visit;
    MIRContext *mir_context_;
    std::string frontier_name;

    VertexDeduplicationVisitor(MIRContext* mir_context): mir_context_(mir_context) {
    }

    virtual void visit(mir::StmtBlock::Ptr) override;
  };
};

}
#endif //FRONTIER_DEDUP_LOWER_H_
