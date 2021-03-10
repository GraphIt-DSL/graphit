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

  struct RoundParamEmitter : public mir::MIRVisitor {
    using mir::MIRVisitor::visit;
    mir::WhileStmt::Ptr current_while_stmt;
    int curr_no = 0;

    virtual void visit(mir::Call::Ptr) override;
  };

  // Separates frontier from vertex level operations (later for generating correct push calls)
 struct SwarmSwitchCaseSeparator: public mir::MIRVisitor {
   using mir::MIRVisitor::visit;
   mir::WhileStmt::Ptr current_while_stmt;
   std::vector<int> swarm_single_level;
   std::vector<int> swarm_frontier_level;
   int idx = 0;

   virtual void visit(mir::VertexSetApplyExpr::Ptr) override;
   virtual void visit(mir::AssignStmt::Ptr) override;
   virtual void visit(mir::VarDecl::Ptr) override;
   virtual void visit(mir::Call::Ptr) override;
   void setup_switch_cases(void);

   // create a blank switch statement with no body.
   mir::SwarmSwitchStmt::Ptr create_blank_switch_stmt(int round, bool is_vertex_level) {
     mir::SwarmSwitchStmt::Ptr switch_stmt = std::make_shared<mir::SwarmSwitchStmt>();
     mir::StmtBlock::Ptr stmt_block = std::make_shared<mir::StmtBlock>();
     switch_stmt->round = round;
     switch_stmt->stmt_block = stmt_block;
     switch_stmt->setMetadata<bool>("is_vertex_level", is_vertex_level);
     return switch_stmt;
   }

   // create a switch statement with the statement of stmt_idx in the while loop
   mir::SwarmSwitchStmt::Ptr convert_to_switch_stmt(int stmt_idx, int round, bool is_vertex_level) {
     mir::Stmt::Ptr stmt = (*(current_while_stmt->body->stmts))[stmt_idx];
     mir::SwarmSwitchStmt::Ptr switch_stmt = std::make_shared<mir::SwarmSwitchStmt>();
     mir::StmtBlock::Ptr stmt_block = std::make_shared<mir::StmtBlock>();
     stmt_block->insertStmtEnd(stmt);
     switch_stmt->round = round;
     switch_stmt->stmt_block = stmt_block;
     switch_stmt->setMetadata<bool>("is_vertex_level", is_vertex_level);
     return switch_stmt;
   }

   // assert whether current_while_stmt should be converted to a bucket queue by looking at
   // whether it should be converted to switch statements and if there are frontier level statements.
   bool is_bucket_queue() {
     return (current_while_stmt->getMetadata<bool>("swarm_switch_convert") && !swarm_frontier_level.empty());
   }

   void insert_single_source_case(int i) {
     swarm_single_level.push_back(i);
   }

   // backfills frontier level operations based on the idxs that did not appear in the single vertex level ops.
   void fill_frontier_stmts() {
     int num_stmts = current_while_stmt->body->stmts->size();
     for (int i = 0; i < num_stmts; i++) {
       if (std::find(swarm_single_level.begin(), swarm_single_level.end(), i) == swarm_single_level.end()) {
         swarm_frontier_level.push_back(i);
       }
     }
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
