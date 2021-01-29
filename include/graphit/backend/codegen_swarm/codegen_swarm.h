#ifndef GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_
#define GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>
#include <iostream>
#include <sstream>
#include <graphit/backend/gen_edge_apply_func_decl.h>
namespace graphit {

class CodeGenSwarmFrontierFinder: public mir::MIRVisitor {
 public:
  CodeGenSwarmFrontierFinder() {};

  std::vector<std::string> swarm_frontier_vars;
  std::map<std::string, mir::VarDecl::Ptr> edgeset_var_map;

  void visit(mir::WhileStmt::Ptr);
  void visit(mir::VarDecl::Ptr);
};


class CodeGenSwarm: public mir::MIRVisitor {
 private:
  CodeGenSwarmFrontierFinder* frontier_finder;

 public:
  CodeGenSwarm(std::ostream &input_oss, MIRContext *mir_context, std::string module_name_):
      oss(input_oss), mir_context_(mir_context), module_name(module_name_) {
    indentLevel = 0;
    frontier_finder = new CodeGenSwarmFrontierFinder();
  }
  int genSwarmCode(void);
  int genIncludeStmts(void);
  int genEdgeSets(void);
  int genSwarmFrontiers();
  int genConstants(void);
  int genMainFunction(void);


  void visitBinaryExpr(mir::BinaryExpr::Ptr, std::string);
  virtual void visit(mir::EdgeSetType::Ptr);
  virtual void visit(mir::ScalarType::Ptr);
  virtual void visit(mir::IntLiteral::Ptr);
  virtual void visit(mir::FloatLiteral::Ptr);
  virtual void visit(mir::BoolLiteral::Ptr);
  virtual void visit(mir::StringLiteral::Ptr);
  virtual void visit(mir::AndExpr::Ptr expr);
  virtual void visit(mir::OrExpr::Ptr expr);
  virtual void visit(mir::XorExpr::Ptr expr);
  virtual void visit(mir::NotExpr::Ptr expr);
  virtual void visit(mir::MulExpr::Ptr expr);
  virtual void visit(mir::DivExpr::Ptr expr);
  virtual void visit(mir::AddExpr::Ptr expr);
  virtual void visit(mir::SubExpr::Ptr expr);
  virtual void visit(mir::VarExpr::Ptr expr);
  virtual void visit(mir::Call::Ptr);
  virtual void visit(mir::ExprStmt::Ptr);
  virtual void visit(mir::EdgeSetLoadExpr::Ptr);
  virtual void visit(mir::ListType::Ptr);
  virtual void visit(mir::ListAllocExpr::Ptr);
  virtual void visit(mir::VertexSetType::Ptr);
  virtual void visit(mir::AssignStmt::Ptr);
  virtual void visit(mir::TensorArrayReadExpr::Ptr);
  virtual void visit(mir::FuncDecl::Ptr);
  virtual void visit(mir::ElementType::Ptr);
  virtual void visit(mir::StmtBlock::Ptr);
  virtual void visit(mir::PrintStmt::Ptr);
  virtual void visit(mir::EqExpr::Ptr);
  virtual void visit(mir::VarDecl::Ptr);
  virtual void visit(mir::WhileStmt::Ptr);
  virtual void visit(mir::ForStmt::Ptr);
  virtual void visit(mir::ReduceStmt::Ptr);
  virtual void visit(mir::VertexSetApplyExpr::Ptr);
  virtual void visit(mir::PushEdgeSetApplyExpr::Ptr);

  virtual void visit(mir::OrderedProcessingOperator::Ptr);
  virtual void visit(mir::PriorityUpdateOperatorMin::Ptr);

  unsigned getIndentLevel() {
    return indentLevel;
  }
 protected:
  void indent() { ++indentLevel; }
  void dedent() { --indentLevel; }
  void printIndent() { oss << std::string(indentLevel, '\t'); }
  std::ostream &oss;
  std::string module_name;
  unsigned      indentLevel;
  MIRContext * mir_context_;
};

// Generates code specific to swarm-convertible while loops that involve frontiers.
class CodeGenSwarmQueueEmitter: public CodeGenSwarm {
 public:
  using CodeGenSwarm::CodeGenSwarm;
  using CodeGenSwarm::visit;
  mir::WhileStmt::Ptr current_while_stmt;
  int stmt_idx = 0;
  bool push_inserted = false;

  int round = 0;

  void setIndentLevel(unsigned level) {
    indentLevel = level;
  }

  int getPushIncrement(int idx) {
    if (!current_while_stmt->getMetadata<bool>("swarm_switch_convert")) return 1;
    std::vector<int> single_idxs = current_while_stmt->getMetadata<std::vector<int>>("swarm_single_level");
    std::vector<int> frontier_idxs = current_while_stmt->getMetadata<std::vector<int>>("swarm_frontier_level");
    int num_stmts = current_while_stmt->body->stmts->size();

    // if the statement is a single vertex stmt
    if (std::find(single_idxs.begin(), single_idxs.end(), idx) != single_idxs.end()) {
      if (single_idxs.size() == 1) return num_stmts;
      int curr_idx = std::find(single_idxs.begin(), single_idxs.end(), idx) - single_idxs.begin();
      int inc = single_idxs[(curr_idx + 1) % single_idxs.size()] - single_idxs[curr_idx];
      return inc < 0 ? inc + num_stmts : inc;
    }

    // if the statement should be evaluated per frontier
    if (frontier_idxs.size() == 1) return num_stmts;
    int curr_idx = std::find(frontier_idxs.begin(), frontier_idxs.end(), idx) - frontier_idxs.begin();
    int inc = frontier_idxs[(curr_idx + 1) % frontier_idxs.size()] - frontier_idxs[curr_idx];
    return inc < 0 ? inc + num_stmts : inc;
  }

  void visit(mir::StmtBlock::Ptr) override;
  void visit(mir::PushEdgeSetApplyExpr::Ptr) override ;
  void visit(mir::VertexSetApplyExpr::Ptr) override;
  void visit(mir::WhileStmt::Ptr) override;
  void visit(mir::VarDecl::Ptr) override;
  void visit(mir::VarExpr::Ptr) override;
  void visit(mir::SwarmSwitchStmt::Ptr);
};

}
#endif //GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_
