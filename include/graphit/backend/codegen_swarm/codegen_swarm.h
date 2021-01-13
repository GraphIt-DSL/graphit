#ifndef GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_
#define GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>
#include <iostream>
#include <sstream>
#include <graphit/backend/gen_edge_apply_func_decl.h>
namespace graphit {

class CodeGenSwarm: mir::MIRVisitor {
 private:
  void indent() { ++indentLevel; }
  void dedent() { --indentLevel; }
  void printIndent() { oss << std::string(indentLevel, '\t'); }
  std::ostream &oss;
  std::string module_name;
  unsigned      indentLevel;
  MIRContext * mir_context_;

 public:
  CodeGenSwarm(std::ostream &input_oss, MIRContext *mir_context, std::string module_name_):
      oss(input_oss), mir_context_(mir_context), module_name(module_name_) {
    indentLevel = 0;
  }
  int genSwarmCode(void);
  int genIncludeStmts(void);
  int genEdgeSets(void);
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
  virtual void visit(mir::VertexSetType::Ptr);
  virtual void visit(mir::AssignStmt::Ptr);
  virtual void visit(mir::TensorArrayReadExpr::Ptr);
  virtual void visit(mir::FuncDecl::Ptr);
  virtual void visit(mir::ElementType::Ptr);
  virtual void visit(mir::StmtBlock::Ptr);
  virtual void visit(mir::PrintStmt::Ptr);
  virtual void visit(mir::VarDecl::Ptr);
  virtual void visit(mir::ForStmt::Ptr);
  virtual void visit(mir::ReduceStmt::Ptr);
  virtual void visit(mir::VertexSetApplyExpr::Ptr);
  virtual void visit(mir::PushEdgeSetApplyExpr::Ptr);

  virtual void visit(mir::OrderedProcessingOperator::Ptr);
  virtual void visit(mir::PriorityUpdateOperatorMin::Ptr);

};



}
#endif //GRAPHIT_INCLUDE_GRAPHIT_BACKEND_CODEGEN_SWARM_CODEGEN_SWARM_H_
