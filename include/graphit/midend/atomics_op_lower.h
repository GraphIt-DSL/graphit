//
// Created by Yunming Zhang on 7/18/17.
//

#ifndef GRAPHIT_ATOMICS_OP_LOWER_H
#define GRAPHIT_ATOMICS_OP_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>


namespace graphit {

    /**
     * Lowers the code in the apply functions to use atomic reduction or CAS
     */
    class AtomicsOperationsLower {

        struct ApplyExprVisitor : public mir::MIRVisitor {
            ApplyExprVisitor(MIRContext *mir_context, Schedule *schedule) :
                    mir_context_(mir_context), schedule_(schedule) {}

            virtual void visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr);

        private:
            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;
        };

        struct ReduceStmtLower : public mir::MIRVisitor {

        };

        struct CompareAndSwapStmtLower : public mir::MIRVisitor {

        };

    };
}

#endif //GRAPHIT_ATOMICS_OP_LOWER_H
