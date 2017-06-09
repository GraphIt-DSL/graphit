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

            };

            // A for loop range domain with integer bounds
            struct RangeDomain {
                int lower;
                int upper;

                typedef std::shared_ptr<RangeDomain> Ptr;

            };

            struct ForStmtNode : StmtNode {
                typedef std::shared_ptr<ForStmt> Ptr;
                RangeDomain::Ptr for_domain_;
                fir::ForStmt::Ptr emitFIRNode();
                // append the stmt block to the body of the current for stmt node
                void appendLoopBody(StmtBlock stmt_block);
            };

//            struct NameNode : StmtNode {
//                typedef std::shared_ptr<NameNode> Ptr;
//
//            };

            struct ProgramNode : public LowLevelScheduleNode {
                typedef std::shared_ptr<ProgramNode> Ptr;

                ProgramNode(graphit::FIRContext fir_context) {
                    fir_program_ = fir_context.getProgram();
                }

                StmtNode cloneLabelLoopBody(std::string label);
                // Inserts a ForStmt node before the label
                bool insertBefore(ForStmtNode for_stmt, std::string label);
                bool insertAfter(ForStmtNode for_stmt, std::string label);

                bool removeLabelNode(std::string label);

            private:
                fir::Program::Ptr fir_program_;
            };


        }
    }
}

#endif //GRAPHIT_LOW_LEVEL_SCHEDULE_H_H
