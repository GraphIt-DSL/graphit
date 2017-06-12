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

            struct StmtNode : public  LowLevelScheduleNode {
                typedef std::shared_ptr<StmtNode> Ptr;

            };

            struct StmtBlockNode : public StmtNode {
                typedef std::shared_ptr<StmtBlockNode> Ptr;

                StmtBlockNode(fir::StmtBlock::Ptr fir_stmt_blk) :
                    fir_stmt_block_(fir_stmt_blk) {};

                int getNumStmts(){
                    if (fir_stmt_block_)
                        return fir_stmt_block_->stmts.size();
                    else
                        return 0;
                }

                fir::StmtBlock::Ptr getFirStmtBlk() {
                    return fir_stmt_block_;
                }

            private:
                fir::StmtBlock::Ptr fir_stmt_block_;
            };

            // A for loop range domain with integer bounds
            struct RangeDomain {
                int lower_;
                int upper_;

                typedef std::shared_ptr<RangeDomain> Ptr;
                RangeDomain(int lower, int upper)
                        : lower_(lower), upper_(upper) {}

            };

            struct ForStmtNode : StmtNode {
                typedef std::shared_ptr<ForStmtNode> Ptr;
                RangeDomain::Ptr for_domain_;

                ForStmtNode(RangeDomain::Ptr range_domain, std::string label)
                        : range_domain_(range_domain), label_(label) {};

                fir::ForStmt::Ptr emitFIRNode();
                // append the stmt block to the body of the current for stmt node
                void appendLoopBody(StmtBlockNode::Ptr stmt_block);

                StmtBlockNode::Ptr getBody(){
                    return body_;
                }

            private:
                std::string label_;
                RangeDomain::Ptr range_domain_;
                StmtBlockNode::Ptr body_;
            };

//            struct NameNode : StmtNode {
//                typedef std::shared_ptr<NameNode> Ptr;
//
//            };

            struct ProgramNode : public LowLevelScheduleNode {
                typedef std::shared_ptr<ProgramNode> Ptr;

                ProgramNode(graphit::FIRContext* fir_context) {
                    fir_program_ = fir_context->getProgram();
                }

                // Clones the body of the loop with the input label
                StmtBlockNode::Ptr cloneLabelLoopBody(std::string label);
                // Inserts a ForStmt node before the label
                bool insertBefore(ForStmtNode::Ptr for_stmt, std::string label);
                bool insertAfter(ForStmtNode::Ptr for_stmt, std::string label);

                bool removeLabelNode(std::string label);

            private:
                fir::Program::Ptr fir_program_;
            };

        }
    }
}

#endif //GRAPHIT_LOW_LEVEL_SCHEDULE_H_H
