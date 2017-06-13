//
// Created by Yunming Zhang on 6/8/17.
//

#include <graphit/frontend/low_level_schedule.h.h>
#include <graphit/frontend/clone_loop_body_visitor.h>

namespace graphit {
    namespace fir {
        namespace low_level_schedule {

            /**
             * Visitor used for inserting a loop body node or name node before a label
             */
            struct InsertBeforeLabelVisitor : public FIRVisitor {

                InsertBeforeLabelVisitor(){
                    input_for_stmt_node_ = nullptr;
                    input_name_node_ = nullptr;
                }

                bool insertBeforeLabel(fir::Program::Ptr fir_program,
                                       ForStmtNode::Ptr for_stmt_node, std::string label){
                    // be default, we assume the input label is not found
                    success_flag_ = false;
                    target_label_ = label;
                    input_for_stmt_node_= for_stmt_node;
                    fir_program->accept(this);
                    return success_flag_;
                }

                bool insertBeforeLabel(fir::Program::Ptr fir_program,
                                       NameNode::Ptr name_node, std::string label){
                    // be default, we assume the input label is not found
                    success_flag_ = false;
                    target_label_ = label;
                    input_name_node_= name_node;
                    fir_program->accept(this);
                    return success_flag_;
                }


                virtual void visit (fir::StmtBlock::Ptr stmt_block){
                    int idx = 0;
                    auto blk_stmts = stmt_block->stmts;
                    for (auto & stmt : blk_stmts) {
                        if(stmt->stmt_label != "")
                        if (label_scope_.tryScope(stmt->stmt_label) == target_label_){
                            success_flag_ = true;

                            if (input_for_stmt_node_ != nullptr){
                                blk_stmts.insert(blk_stmts.begin() + idx,
                                                 input_for_stmt_node_->emitFIRNode());
                            }

                            if (input_name_node_ != nullptr){
                                blk_stmts.insert(blk_stmts.begin() + idx,
                                                 input_name_node_->emitFIRNode());
                            }

                            break;
                        }
                        idx++;
                    }

                    stmt_block->stmts = blk_stmts;
                }

            private:
                std::string target_label_;
                ForStmtNode::Ptr input_for_stmt_node_;
                NameNode::Ptr input_name_node_;
                bool success_flag_;
            };


            /**
            * Visitor used for remove a node with a certain label
            */
            struct RemoveLabelVisitor : public FIRVisitor {

                RemoveLabelVisitor(){

                }

                bool removeLabel(fir::Program::Ptr fir_program, std::string label){
                    // be default, we assume the input label is not found
                    success_flag_ = false;
                    target_label_ = label;
                    fir_program->accept(this);
                    return success_flag_;
                }


                virtual void visit (fir::StmtBlock::Ptr stmt_block){
                    int idx = 0;
                    auto blk_stmts = stmt_block->stmts;
                    for (auto & stmt : blk_stmts) {
                        if(stmt->stmt_label != "")
                            if (label_scope_.tryScope(stmt->stmt_label) == target_label_){
                                success_flag_ = true;

                                blk_stmts.erase(blk_stmts.begin() + idx);

                                break;
                            }
                        idx++;
                    }

                    stmt_block->stmts = blk_stmts;
                }

            private:
                std::string target_label_;
                bool success_flag_;
            };


            StmtBlockNode::Ptr ProgramNode::cloneLabelLoopBody(std::string label) {

                // Traverse the fir::program nodes and copy the labeled node body
                auto clone_loop_body_visitor = CloneLoopBodyVisitor();
                fir::StmtBlock::Ptr fir_stmt_blk = clone_loop_body_visitor.CloneLoopBody(fir_program_, label);

                auto stmt_blk_node = std::make_shared<StmtBlockNode>(fir_stmt_blk);

                return stmt_blk_node;
            }

            bool ProgramNode::insertAfter(ForStmtNode::Ptr for_stmt, std::string label) {
                return true;
            }

            bool ProgramNode::insertAfter(NameNode::Ptr for_stmt, std::string label) {
                return true;
            }

            bool ProgramNode::insertBefore(NameNode::Ptr name_node, std::string label) {
                auto insert_before_visitor = InsertBeforeLabelVisitor();
                return insert_before_visitor.insertBeforeLabel(fir_program_, name_node, label);
            }

            bool ProgramNode::insertBefore(ForStmtNode::Ptr for_stmt, std::string label) {
                auto insert_before_visitor = InsertBeforeLabelVisitor();
                return insert_before_visitor.insertBeforeLabel(fir_program_, for_stmt, label);
            }

            bool ProgramNode::removeLabelNode(std::string label) {
                auto remove_label_visitor = RemoveLabelVisitor();
                remove_label_visitor.removeLabel(fir_program_, label);
                return true;
            }

            fir::ForStmt::Ptr ForStmtNode::emitFIRNode() {
                auto fir_stmt = std::make_shared<fir::ForStmt>();
                fir_stmt->stmt_label = label_;
                fir_stmt->body = body_->emitFIRNode();
                fir_stmt->domain = range_domain_->emitFIRRangeDomain();

                auto fir_identifier = std::make_shared<fir::Identifier>();
                fir_identifier->ident = loop_var_;

                fir_stmt->loopVar = fir_identifier;
                return fir_stmt;
            }

            void ForStmtNode::appendLoopBody(StmtBlockNode::Ptr stmt_block) {
                if (body_ == nullptr)
                    body_ = stmt_block;
                else
                    body_->appendStmtBlockNode(stmt_block);
            }

            fir::NameNode::Ptr NameNode::emitFIRNode() {
                auto fir_name_node = std::make_shared<fir::NameNode>();
                fir_name_node->body = body_->emitFIRNode();
                fir_name_node->stmt_label = label_;
                return fir_name_node;
            }

            void StmtBlockNode::appendStmtBlockNode(StmtBlockNode::Ptr stmt_block) {

                auto other_stmt_blk = stmt_block->emitFIRNode()->stmts;

                fir_stmt_block_->stmts.insert(fir_stmt_block_->stmts.end(),
                                              other_stmt_blk.begin(), other_stmt_blk.end());
            }
        }
    }
}