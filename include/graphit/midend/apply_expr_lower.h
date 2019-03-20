//
// Created by Yunming Zhang on 5/30/17.
//

#ifndef GRAPHIT_APPLY_EXPR_LOWER_H
#define GRAPHIT_APPLY_EXPR_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {
    class ApplyExprLower {
    public:
        // construct with no input schedule
        ApplyExprLower(MIRContext *mir_context) : mir_context_(mir_context) {

        }

        //constructor with input schedule
        ApplyExprLower(MIRContext *mir_context, Schedule *schedule)
                : schedule_(schedule), mir_context_(mir_context) {};


        void lower();

        //mir rewriter for rewriting the abstract tensor reads into the right type of reads
        struct LowerApplyExpr : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerApplyExpr(Schedule* schedule, MIRContext* mir_context)
                    : schedule_(schedule), mir_context_(mir_context){

            };

            //Lowers edgeset apply expressions
            virtual void visit(mir::EdgeSetApplyExpr::Ptr edgeset_apply_expr);
            virtual void visit(mir::VertexSetApplyExpr::Ptr vertexset_apply_expr);


            Schedule * schedule_;
            MIRContext* mir_context_;
        };

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;
    };
}

#endif //GRAPHIT_APPLY_EXPR_LOWER_H
