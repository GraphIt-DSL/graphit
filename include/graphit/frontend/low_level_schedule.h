//
// Created by Yunming Zhang on 6/8/17.
//

#ifndef GRAPHIT_LOW_LEVEL_SCHEDULE_H_H
#define GRAPHIT_LOW_LEVEL_SCHEDULE_H_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>
#include "fir_context.h"
#include "clone_apply_node_visitor.h"
#include "clone_for_stmt_node_visitor.h"

/**
 * This file contains the classes and API calss for the low level schedule language API
 * These classes and method calls are used in high level schedule APIs
 */
namespace graphit {
    namespace fir {
        namespace low_level_schedule {

            struct LowLevelScheduleNode
                    : public std::enable_shared_from_this<LowLevelScheduleNode> {
                typedef std::shared_ptr<LowLevelScheduleNode> Ptr;

            };

            struct StmtNode : public LowLevelScheduleNode {
                typedef std::shared_ptr<StmtNode> Ptr;
            };

            struct StmtBlockNode : public StmtNode {
                typedef std::shared_ptr<StmtBlockNode> Ptr;

                StmtBlockNode(fir::StmtBlock::Ptr fir_stmt_blk) :
                        fir_stmt_block_(fir_stmt_blk) {};

                StmtBlockNode()  {
                    fir_stmt_block_ = std::make_shared<fir::StmtBlock>();
                };

                int getNumStmts() {
                    if (fir_stmt_block_)
                        return fir_stmt_block_->stmts.size();
                    else
                        return 0;
                }

                void appendStmtBlockNode(StmtBlockNode::Ptr stmt_block);
                void appendFirStmt(fir::Stmt::Ptr fir_stmt);

                fir::StmtBlock::Ptr emitFIRNode() {
                    return fir_stmt_block_;
                }

            private:
                fir::StmtBlock::Ptr fir_stmt_block_;
            };

            // A for loop range domain with integer bounds
            struct RangeDomain {

                typedef std::shared_ptr<RangeDomain> Ptr;

                RangeDomain(fir::RangeDomain::Ptr fir_range_domain)
                {
                    lower_ = fir::to<fir::IntLiteral>(fir_range_domain->lower)->val;
                    upper_ = fir::to<fir::IntLiteral>(fir_range_domain->upper)->val;
                }

                RangeDomain(int lower, int upper)
                        : lower_(lower), upper_(upper) {}

                fir::RangeDomain::Ptr emitFIRRangeDomain() {
                    fir::RangeDomain::Ptr fir_range_domain = std::make_shared<fir::RangeDomain>();
                    auto lower_expr = std::make_shared<fir::IntLiteral>();
                    lower_expr->val = lower_;
                    auto upper_expr = std::make_shared<fir::IntLiteral>();
                    upper_expr->val = upper_;
                    fir_range_domain->lower = lower_expr;
                    fir_range_domain->upper = upper_expr;
                    return fir_range_domain;
                }

            private:
                int lower_;
                int upper_;

            };

            struct ForStmtNode : StmtNode {
                typedef std::shared_ptr<ForStmtNode> Ptr;
                RangeDomain::Ptr for_domain_;

                ForStmtNode(RangeDomain::Ptr range_domain, std::string label)
                        : range_domain_(range_domain), label_(label) {};

                ForStmtNode(RangeDomain::Ptr range_domain,
                            StmtBlockNode::Ptr body,
                            std::string label,
                            std::string loop_var)
                        : range_domain_(range_domain),
                          label_(label),
                          body_(body),
                          for_domain_(range_domain),
                          loop_var_(loop_var) {};

                ForStmtNode(fir::ForStmt::Ptr for_stmt, std::string stmt_label)
                {
                    fir::RangeDomain::Ptr fir_domain_range;
                    if (fir::isa<fir::RangeDomain>(for_stmt->domain))
                        fir_domain_range = fir::to<fir::RangeDomain>(for_stmt->domain);
                    else
                        std::cout << "error in cloneForStmtNode, not range domain";

                    fir::low_level_schedule::RangeDomain::Ptr schedule_domain_range =
                            std::make_shared<fir::low_level_schedule::RangeDomain>(
                                    fir::to<fir::IntLiteral>(fir_domain_range->lower)->val,
                                    fir::to<fir::IntLiteral>(fir_domain_range->upper)->val);

                    fir::low_level_schedule::StmtBlockNode::Ptr schedule_body_stmt =
                            std::make_shared<fir::low_level_schedule::StmtBlockNode>(for_stmt->body);

                    label_ = stmt_label;
                    body_ = schedule_body_stmt;
                    loop_var_ = for_stmt->loopVar->ident;
                    for_domain_ = schedule_domain_range;
                }

                fir::ForStmt::Ptr emitFIRNode();

                // append the stmt block to the body of the current for stmt node
                void appendLoopBody(StmtBlockNode::Ptr stmt_block);

                StmtBlockNode::Ptr getBody() {
                    return body_;
                }

            private:
                std::string label_;
                RangeDomain::Ptr range_domain_;
                StmtBlockNode::Ptr body_;
                std::string loop_var_;
            };

            struct NameNode : StmtNode {
                typedef std::shared_ptr<NameNode> Ptr;

                NameNode(StmtBlockNode::Ptr stmt_block, std::string label)
                        : body_(stmt_block), label_(label) {}

                fir::NameNode::Ptr emitFIRNode();

                StmtBlockNode::Ptr getBody() {
                    return body_;
                }

            private:
                StmtBlockNode::Ptr body_;
                std::string label_;
            };

            struct FuncDeclNode : public LowLevelScheduleNode {
                // constructs a scheduling func decl node by cloning a decl from func
                FuncDeclNode(fir::FuncDecl::Ptr fir_func_decl) {
                    fir_func_decl_ = fir_func_decl;
                }

                void setFunctionName(std::string func_name){
                    if (fir_func_decl_ != nullptr)
                        fir_func_decl_->name->ident = func_name;
                    else
                        std::cout << "error in setting function name, nullptr" << std::endl;
                }

                std::string getFunctionName(){
                    if (fir_func_decl_ != nullptr)
                        return fir_func_decl_->name->ident;
                    else
                        std::cout << "error in setting function name, nullptr" << std::endl;
                        return "";
                }

                StmtBlockNode::Ptr getBody() {
                    if (fir_func_decl_->body)
                        return std::make_shared<StmtBlockNode>(StmtBlockNode(fir_func_decl_->body));
                    else
                        return nullptr;
                }

                typedef std::shared_ptr<FuncDeclNode> Ptr;

                void appendFuncDeclBody(StmtBlockNode::Ptr func_decl_body);


                fir::FuncDecl::Ptr emitFIRNode() {
                    return fir_func_decl_;
                };
            private:
                fir::FuncDecl::Ptr fir_func_decl_;

            };

            //the low level scheduling language works with expr stmt apply expressions
            struct ApplyNode : StmtNode {
                typedef std::shared_ptr<ApplyNode> Ptr;
                ApplyNode(fir::ExprStmt::Ptr apply_expr_stmt) : apply_expr_stmt_(apply_expr_stmt){};
                //updates the label of the fir::ExprStmt node
                void updateStmtLabel(std::string label);
                void updateApplyFunc(std::string new_apply_func_name);
                std::string getApplyFuncName();
                fir::ExprStmt::Ptr emitFIRNode();
            private:
                fir::ExprStmt::Ptr apply_expr_stmt_;
            };

            struct ProgramNode : public LowLevelScheduleNode {
                typedef std::shared_ptr<ProgramNode> Ptr;

                ProgramNode(graphit::FIRContext *fir_context) {
                    fir_program_ = fir_context->getProgram();
                }

                // Clones the body of the loop with the input label
                StmtBlockNode::Ptr cloneLabelLoopBody(std::string label);

                // Inserts a ForStmt node before and after a label
                bool insertBefore(ForStmtNode::Ptr for_stmt, std::string label);

                //may be we can just get around everything with insert before
                // (insert before the node and remove the node is the same as insert after and remove)
                //bool insertAfter(ForStmtNode::Ptr for_stmt, std::string label);

                // Inserts a name node before and after a label
                bool insertBefore(NameNode::Ptr for_stmt, std::string label);

                //bool insertAfter(NameNode::Ptr for_stmt, std::string label);

                bool insertBefore(ApplyNode::Ptr apply_node, std::string label);
                bool replaceLabel(ApplyNode::Ptr apply_node, std::string label);

                // Removes a statement associated with the label
                bool removeLabelNode(std::string label);

                void insertAfter(FuncDeclNode::Ptr func_decl_node, std::string function_name);

                StmtBlockNode::Ptr cloneFuncBody(std::string func_name);
                FuncDeclNode::Ptr cloneFuncDecl(std::string func_name);
                ApplyNode::Ptr cloneApplyNode(std::string stmt_label);
                ForStmtNode::Ptr cloneForStmtNode(std::string for_stmt_label);

            private:
                fir::Program::Ptr fir_program_;
            };



        }
    }
}

#endif //GRAPHIT_LOW_LEVEL_SCHEDULE_H_H
