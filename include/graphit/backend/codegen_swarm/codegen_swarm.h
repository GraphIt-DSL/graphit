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

  std::vector<std::string> swarm_frontier_prioq_vars;
  std::vector<std::string> swarm_frontier_bucketq_vars;
  std::map<std::string, mir::VarDecl::Ptr> edgeset_var_map;

  // helper to check if frontier is in either of the swarm frontier vectors.
  bool isa_frontier(std::string frontier_name) {
    if (std::find(swarm_frontier_prioq_vars.begin(), swarm_frontier_prioq_vars.end(), frontier_name) != swarm_frontier_prioq_vars.end()) {
      return true;
    }
    return std::find(swarm_frontier_bucketq_vars.begin(), swarm_frontier_bucketq_vars.end(), frontier_name)
        != swarm_frontier_bucketq_vars.end();
  }

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
  virtual void visit(mir::VertexSetAllocExpr::Ptr);
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
  bool is_insert_call = false;

  int round = 0;

  // prints the codegen for the src vertex
  void printSrcVertex() {
    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      oss << "std::get<0>(src_pair)";
    } else {
      oss << "src";
    }
  }

  // prints a push statement, where the push increment is 1, and there is an option to increment
  // a frontier round, and an option to push a different vertex (dst or src)
  void printPushStatement(bool increment_round, bool same_vertex, std::string next_vertex="") {
    printIndent();
    oss << "push(level + 1, ";
    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      if(current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size() > 2) {
        assert(false && "more passed variables (> 2) not supported yet");
      }
    }

    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      oss << "std::make_pair<int, ";
      for (int i = 0; i < current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size(); i++) {
        auto add_var = current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars")[i];
        add_var.getType()->accept(this);
        if (i < current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size() - 1) {
          oss << ", ";
        }
      }
      oss << ">(";
      oss << (same_vertex ? "std::get<0>(src_pair)" : next_vertex) << ", std::get<1>(src_pair)";
      if (increment_round) oss << "+ 1";
      oss << ")";
    } else {
      oss << (same_vertex ? "src" : next_vertex);
    }
    oss << ");" << std::endl;
  }

  bool is_bucket_queue() {
    return (current_while_stmt->getMetadata<bool>("swarm_switch_convert")
        && current_while_stmt->hasMetadata<mir::StmtBlock::Ptr>("new_frontier_bucket"));
  }

  void setIndentLevel(unsigned level) {
    indentLevel = level;
  }

  void visit(mir::StmtBlock::Ptr) override;
  void visit(mir::PushEdgeSetApplyExpr::Ptr) override ;
  void visit(mir::VertexSetApplyExpr::Ptr) override;
  void visit(mir::WhileStmt::Ptr) override;
  void visit(mir::VarDecl::Ptr) override;
  void visit(mir::VarExpr::Ptr) override;
  void visit(mir::VertexSetType::Ptr) override;
  void visit(mir::SwarmSwitchStmt::Ptr);
  void visit(mir::AssignStmt::Ptr) override;
  void visit(mir::FuncDecl::Ptr) override;
  void visit(mir::Call::Ptr) override;
};

}
#endif //GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_
