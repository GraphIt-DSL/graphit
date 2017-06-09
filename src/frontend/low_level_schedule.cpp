//
// Created by Yunming Zhang on 6/8/17.
//

#include <graphit/frontend/low_level_schedule.h.h>

namespace graphit {
    namespace fir {
        namespace low_level_schedule {

            StmtNode ProgramNode::cloneLabelLoopBody(std::string label) {

            }

            bool ProgramNode::insertAfter(ForStmtNode for_stmt, std::string label) {

            }

            bool ProgramNode::insertBefore(ForStmtNode for_stmt, std::string label) {

            }

            bool ProgramNode::removeLabelNode(std::string label) {

            }

            fir::ForStmt::Ptr ForStmtNode::emitFIRNode() {
                auto fir_stmt = std::make_shared<fir::ForStmt>();
                return fir_stmt;
            }

            void ForStmtNode::appendLoopBody(StmtBlock stmt_block) {

            }
        }
    }
}