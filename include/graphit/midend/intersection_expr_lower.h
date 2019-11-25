//
// Created by Tugsbayasgalan Manlaibaatar on 2019-10-08.
//

#ifndef GRAPHIT_INTERSECTION_EXPR_LOWER_H
#define GRAPHIT_INTERSECTION_EXPR_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {
    class IntersectionExprLower {
    public:
        // construct with no input schedule
        IntersectionExprLower(MIRContext *mir_context) : mir_context_(mir_context) {

        }

        //constructor with input schedule
        IntersectionExprLower(MIRContext *mir_context, Schedule *schedule)
        : schedule_(schedule), mir_context_(mir_context) {};


        void lower();

        struct LowerIntersectionExpr : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerIntersectionExpr(Schedule* schedule, MIRContext* mir_context)
                    : schedule_(schedule), mir_context_(mir_context){

            };

            virtual void visit(mir::IntersectionExpr::Ptr intersection_expr);
            virtual void visit(mir::IntersectNeighborExpr::Ptr intersection_expr);


            Schedule * schedule_;
            MIRContext* mir_context_;
        };


    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;
    };
}

#endif //GRAPHIT_INTERSECTION_EXPR_LOWER_H