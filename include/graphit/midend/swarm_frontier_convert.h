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
    std::vector<int> insert_stmt_idxs; // where in the while stmt body the insert vertex calls are.
    std::vector<mir::Expr::Ptr> insert_stmt_frontier_list_vars; // the VarExpr representing the VFL for each insert_vertex call.
    std::vector<mir::Var> insert_stmt_incr_vars; // The Var with the variable representing the frontier list round.
    bool insert_found = false;
    int curr_no = 0;

    virtual void visit(mir::Call::Ptr) override;
    virtual void visit(mir::StmtBlock::Ptr) override;
   
   // Modify while stmt body by adding an update size call to frontier lists after ach frontier list insert call. 
    void fill_update_size_stmts() {
	    if (insert_stmt_idxs.size() == 0) return;
	    int insert_stmt_idx = 0;
	    auto body = current_while_stmt->body;
	    std::vector<mir::Stmt::Ptr> new_stmts;
	
	// Insert update_size calls after each insert_vertex statement, whose idx's are indicated in the insert_stmt_idxs vector.
        for (int i = 0; i < body->stmts->size() ; i++) {
		new_stmts.push_back((*(body->stmts))[i]);
		if (i == insert_stmt_idxs[insert_stmt_idx]) {
			mir::Call::Ptr new_call = std::make_shared<mir::Call>();
			new_call->name = "builtin_update_size";
			auto vfl_varexpr = insert_stmt_frontier_list_vars[insert_stmt_idx];
			new_call->args.push_back(vfl_varexpr);
			new_call->setMetadata<mir::Var>("increment_round_var", insert_stmt_incr_vars[insert_stmt_idx]);
			auto new_expr_stmt = std::make_shared<mir::ExprStmt>();
			new_expr_stmt->expr = new_call;
			new_stmts.push_back(new_expr_stmt);
			insert_stmt_idx++;
		}
	}
	(*(body->stmts)) = new_stmts;
    }
  };

  struct GlobalVariableFinder: public mir::MIRVisitor {
    using mir::MIRVisitor::visit;
    mir::WhileStmt::Ptr current_while_stmt;
    std::vector<std::string> declared_vars;
    std::vector<mir::Var> global_vars;
    std::string frontier_name;

    virtual void visit(mir::Call::Ptr);
    virtual void visit(mir::AssignStmt::Ptr);
    virtual void visit(mir::VarDecl::Ptr);
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
