//
// Created by Yunming Zhang on 6/29/17.
//

#ifndef GRAPHIT_CHANGE_TRACKING_LOWER_H
#define GRAPHIT_CHANGE_TRACKING_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {

    class ChangeTrackingLower {
    public:
        ChangeTrackingLower(MIRContext *mir_context, Schedule *schedule)
                : schedule_(schedule), mir_context_(mir_context) {};

        void lower();

        struct ApplyExprVisitor : public mir::MIRVisitor {

            ApplyExprVisitor(MIRContext *mir_context, Schedule *schedule) :
                    mir_context_(mir_context), schedule_(schedule) {}

            virtual void visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr);

        private:
            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;

            void insertSerialReturnStmtForTrackingChange(mir::FuncDecl::Ptr apply_func_decl, std::string tracking_field);
        };

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;

    };
}

#endif //GRAPHIT_CHANGE_TRACKING_LOWER_H
