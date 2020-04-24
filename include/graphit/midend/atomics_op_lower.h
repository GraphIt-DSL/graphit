//
// Created by Yunming Zhang on 7/18/17.
//

#ifndef GRAPHIT_ATOMICS_OP_LOWER_H
#define GRAPHIT_ATOMICS_OP_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir.h>

namespace graphit {

    /**
     * Lowers the code in the apply functions to use atomic reduction or CAS
     */
    class AtomicsOpLower {

    public:
        AtomicsOpLower(MIRContext *mir_context) : mir_context_(mir_context) {};

        struct ApplyExprVisitor : public mir::MIRVisitor {
            ApplyExprVisitor(MIRContext *mir_context) :
                    mir_context_(mir_context){}

            virtual void visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::HybridDenseForwardEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr apply_expr);

        private:

            //does a pattern recognition and replace if condition and assignment in apply function with CAS in apply funciton
            //bool lowerCompareAndSwap(std::string to_func, std::string from_func, std::string apply_func, mir::EdgeSetApplyExpr::Ptr apply_expr);
            bool lowerCompareAndSwap(mir::FuncExpr::Ptr to_func, mir::FuncExpr::Ptr from_func, mir::FuncExpr::Ptr apply_func, mir::EdgeSetApplyExpr::Ptr apply_expr);

            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;

            //generates atomics for push, pull, hybrid denseforward
            void singleFunctionEdgeSetApplyExprAtomicsLower(mir::EdgeSetApplyExpr::Ptr apply_expr);
        };

        struct ReduceStmtLower : public mir::MIRVisitor {
            ReduceStmtLower(MIRContext* mir_context) : mir_context_(mir_context){
            }


            virtual void visit(mir::ReduceStmt::Ptr reduce_stmt);

        private:
            MIRContext *mir_context_ = nullptr;

        };

        void lower();


    private:
        MIRContext *mir_context_ = nullptr;



    };
}

#endif //GRAPHIT_ATOMICS_OP_LOWER_H
