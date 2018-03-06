//
// Created by Sherry Yang on 2/27/18.
//

#ifndef GRAPHIT_MERGE_REDUCE_LOWER_H
#define GRAPHIT_MERGE_REDUCE_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {
    class MergeReduceLower {
    public:
        MergeReduceLower(MIRContext *mir_context, Schedule *schedule)
                : mir_context_(mir_context), schedule_(schedule){}

        void lower();

        struct ReduceStmtVisitor : public mir::MIRVisitor {
            ReduceStmtVisitor(MIRContext *mir_context)
                    : mir_context_(mir_context){}

            virtual void visit(mir::ReduceStmt::Ptr reduce_stmt);

            std::string merge_field = "";
            mir::ReduceStmt::ReductionOp reduce_op;

        private:
            MIRContext *mir_context_ = nullptr;
        };

        struct ApplyExprVisitor : public mir::MIRVisitor {

            ApplyExprVisitor(MIRContext *mir_context, Schedule *schedule) :
                    mir_context_(mir_context), schedule_(schedule){}

            virtual void visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr);

        private:
            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;
            void processMergeReduce(mir::EdgeSetApplyExpr::Ptr apply_expr);
        };

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;
    };
}

#endif //GRAPHIT_MERGE_REDUCE_LOWER_H
