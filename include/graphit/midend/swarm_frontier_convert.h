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

  // figures out whether a while loop can be converted to the switch case format for swarm parallelization.
  struct SwitchWhileCaseFinder: public mir::MIRVisitor {
     using mir::MIRVisitor::visit;
     bool can_switch = true;
     std::string frontier_name;

     virtual void visit(mir::EdgeSetApplyExpr::Ptr) override;
     virtual void visit(mir::VertexSetApplyExpr::Ptr) override;
     virtual void visit(mir::AssignStmt::Ptr) override;
     virtual void visit(mir::VarDecl::Ptr) override;

   };

  struct RuntimeInsertConvert : public mir::MIRVisitor {
    mir::StmtBlock::Ptr new_stmt_block = std::make_shared<mir::StmtBlock>();
    using mir::MIRVisitor::visit;

    virtual void visit(mir::StmtBlock::Ptr) override;
    virtual void visit(mir::Call::Ptr) override;
  };

  // Separates frontier from vertex level operations (later for generating correct push calls)
 struct SwarmSwitchCaseSeparator: public mir::MIRVisitor {
   using mir::MIRVisitor::visit;
   mir::WhileStmt::Ptr current_while_stmt;
   int idx = 0;

   virtual void visit(mir::VertexSetApplyExpr::Ptr) override;
   virtual void visit(mir::AssignStmt::Ptr) override;
   virtual void visit(mir::VarDecl::Ptr) override;
   virtual void visit(mir::Call::Ptr) override;

   void insert_single_source_case(int i) {
     std::vector<int> single;
     if (current_while_stmt->hasMetadata<std::vector<int>>("swarm_single_level")) {
       single = current_while_stmt->getMetadata<std::vector<int>>("swarm_single_level");
     }
     single.push_back(i);
     current_while_stmt->setMetadata<std::vector<int>>("swarm_single_level", single);
   }

   // backfills frontier level operations based on the idxs that did not appear in the single vertex level ops.
   void fill_frontier_stmts() {
     std::vector<int> frontier_idxs;
     int num_stmts = current_while_stmt->body->stmts->size();
     std::vector<int> single_idxs = current_while_stmt->getMetadata<std::vector<int>>("swarm_single_level");
     for (int i = 0; i < num_stmts; i++) {
       if (std::find(single_idxs.begin(), single_idxs.end(), i) == single_idxs.end()) {
         frontier_idxs.push_back(i);
       }
     }
     current_while_stmt->setMetadata<std::vector<int>>("swarm_frontier_level", frontier_idxs);
   }
 };

  void analyze(void);
  protected:
    virtual void visit(mir::WhileStmt::Ptr);
  private:
    MIRContext *mir_context_ = nullptr;
};

}
#endif //GRAPHIT_INCLUDE_GRAPHIT_MIDEND_SWARM_FRONTIER_CONVERT_H_
