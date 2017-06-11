//
// Created by Yunming Zhang on 6/8/17.
//

#include <graphit/frontend/low_level_schedule.h.h>
#include <graphit/frontend/clone_loop_body_visitor.h>

namespace graphit {
    namespace fir {
        namespace low_level_schedule {

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

            bool ProgramNode::insertBefore(ForStmtNode::Ptr for_stmt, std::string label) {
                return true;
            }

            bool ProgramNode::removeLabelNode(std::string label) {
                return true;
            }

            fir::ForStmt::Ptr ForStmtNode::emitFIRNode() {
                auto fir_stmt = std::make_shared<fir::ForStmt>();
                return fir_stmt;
            }

            void ForStmtNode::appendLoopBody(StmtBlockNode::Ptr stmt_block) {

            }
        }
    }
}