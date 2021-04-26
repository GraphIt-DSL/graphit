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
  enum QueueType {
    UNORDEREDQUEUE,
    BUCKETQUEUE,
    PRIOQUEUE
  };
  
  struct frontier {
    std::string name;
    QueueType queue_type;
  };

  CodeGenSwarmFrontierFinder() {};

  std::map<std::string, mir::VarDecl::Ptr> edgeset_var_map;
  std::vector<frontier> swarm_frontier_vars;

  // helper to check if frontier is in either of the swarm frontier vectors.
  bool isa_frontier(std::string frontier_name) {
    for (auto f : swarm_frontier_vars) {
      if (f.name == frontier_name) {
        return true;
      }
    }
    return false;
  }

  std::string get_queue_string(std::string frontier_name) {
    for (auto f : swarm_frontier_vars) {
      if (f.name == frontier_name) {
        if (f.queue_type == QueueType::BUCKETQUEUE) {
	  return "BucketQueue";
	}
	if (f.queue_type == QueueType::PRIOQUEUE) {
	  return "PrioQueue";
	}
	if (f.queue_type == QueueType::UNORDEREDQUEUE) {
	  return "UnorderedQueue";
	}
      }
    }
    return "";
  }

  void visit(mir::WhileStmt::Ptr);
  void visit(mir::VarDecl::Ptr);
};

class CodeGenSwarmDedupFinder: public mir::MIRVisitor {
 public:
  CodeGenSwarmDedupFinder() {};
  std::vector<mir::Var> vector_vars;
  std::vector<mir::Expr::Ptr> vector_size_calls;

  void visit(mir::PushEdgeSetApplyExpr::Ptr);

  std::vector<mir::Var> get_vector_vars() {
  	return vector_vars;
  }

  std::vector<mir::Expr::Ptr> get_vector_size_calls() {
  	return vector_size_calls;
  }
};


class CodeGenSwarm: public mir::MIRVisitor {
 private:
  CodeGenSwarmFrontierFinder* frontier_finder;
  CodeGenSwarmDedupFinder* dedup_finder;

 public:
  CodeGenSwarm(std::ostream &input_oss, MIRContext *mir_context, std::string module_name_):
      oss(input_oss), mir_context_(mir_context), module_name(module_name_) {
    indentLevel = 0;
    frontier_finder = new CodeGenSwarmFrontierFinder();
    dedup_finder = new CodeGenSwarmDedupFinder();
  }
  int genSwarmCode(void);
  int genIncludeStmts(void);
  int genEdgeSets(void);
//  int genSwarmFrontiers();
  int genDedupVectors(void);
  int genConstants(void);
  int genMainFunction(void);
  int genSwarmStructs(void);

  void printSpatialHint(mir::Expr::Ptr);
  void printSpatialHint(void);
  void inlineFunction(mir::FuncDecl::Ptr, std::string);
  
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
  virtual void visit(mir::NegExpr::Ptr expr);
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
  virtual void visit(mir::IfStmt::Ptr);
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
  virtual void visit(mir::EnqueueVertex::Ptr) override;

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
 private:

 public:
  enum QueueType {
  	PRIOQUEUE,
	BUCKETQUEUE
  };

  using CodeGenSwarm::CodeGenSwarm;
  using CodeGenSwarm::visit;

  CodeGenSwarmDedupFinder* dedup_finder;
  mir::WhileStmt::Ptr current_while_stmt;
  int stmt_idx = 0;
  bool push_inserted = false;
  bool is_insert_call = false;
  QueueType swarm_queue_type = QueueType::PRIOQUEUE;

  // sets queue type to some QueueType. Probably can later use a scheduling parameter to change this
  void setQueueType(QueueType queue_type) {
  	swarm_queue_type = queue_type;
  }

  // prints the codegen for the src vertex
  void printSrcVertex() {
    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      // this is always src inside the struct.
      oss << "src_struct.src"; 
    } else {
      oss << "src";
    }
  }

  void printRoundIncrement(std::string var_name) {
    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      // check that there are additional arguments in this while stmt struct.
      oss << "src_struct." << current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName() << "_" << var_name;
    } else {
      assert (false && "Don't get here please");
    }
  }

  // prints a push statement, where the push increment is 1, and there is an option to increment
  // a frontier round, and an option to push a different vertex (dst or src)
  void printPushStatement(bool increment_round, bool same_vertex, std::string next_vertex="") {
	  std::string frontier_name = current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName();
	  printIndent();
    oss << "push(level + 1, ";
    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      if(current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size() > 2) {
        assert(false && "more passed variables (> 2) not supported yet");
      }
    }

    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      std::vector<mir::Var> add_src_vars = current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars");
      oss << frontier_name << "_struct{";
      
      oss << (same_vertex ? "src_struct.src, " : next_vertex + ", ");
      for (int i = 0 ; i < add_src_vars.size(); i++) {
	auto add_src_var = add_src_vars[i];
        oss << "src_struct." << frontier_name << "_" << add_src_var.getName();
	if (i != add_src_vars.size() - 1) oss << ", ";
      }
      // i think this is hard-coded to assume increment round is the last elemn in the struct
      if (increment_round) oss << " + 1";
      oss << "}";
    } else {
      oss << (same_vertex ? "src" : next_vertex);
    }
    oss << ");" << std::endl;
  }
  
  // prints a push statement, where the push increment is 1, and there is an option to increment
  // a frontier round, and an option to push a different expr
  void printPushStatement(bool increment_round, bool same_vertex, mir::Expr::Ptr next_vertex_expr) {
	  std::string frontier_name = current_while_stmt->getMetadata<mir::Var>("swarm_frontier_var").getName();
	  printIndent();
    oss << "push(level + 1, ";
    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      if(current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars").size() > 2) {
        assert(false && "more passed variables (> 2) not supported yet");
      }
    }

    if (current_while_stmt->hasMetadata<std::vector<mir::Var>>("add_src_vars")) {
      std::vector<mir::Var> add_src_vars = current_while_stmt->getMetadata<std::vector<mir::Var>>("add_src_vars");
      oss << frontier_name << "_struct{";
      next_vertex_expr->accept(this);
      for (int i = 0 ; i < add_src_vars.size(); i++) {
	auto add_src_var = add_src_vars[i];
        oss << "src_struct." << frontier_name << "_" << add_src_var.getName();
	if (i != add_src_vars.size() - 1) oss << ", ";
      }
      // i think this is hard-coded to assume increment round is the last elemn in the struct
      if (increment_round) oss << " + 1";
      oss << "}";
    } else {
      next_vertex_expr->accept(this);
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
  void visit(mir::PriorityUpdateOperatorMin::Ptr) override;
  void visit(mir::EnqueueVertex::Ptr) override;
  void prepForEachPrioCall(mir::WhileStmt::Ptr);
  void cleanUpForEachPrioCall(mir::WhileStmt::Ptr);
};

}
#endif //GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_
