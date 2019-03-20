//
// Created by Yunming Zhang on 3/20/19.
//

#ifndef GRAPHIT_PRIORITY_FEATURES_LOWERING_H
#define GRAPHIT_PRIORITY_FEATURES_LOWERING_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>


namespace graphit {

    class PriorityFeaturesLower {
    public:

        PriorityFeaturesLower(MIRContext *mir_context) : mir_context_(mir_context) {};

        PriorityFeaturesLower(MIRContext *mir_context, Schedule *schedule)
                : schedule_(schedule), mir_context_(mir_context) {};


        // use the schedule to set the type for MIR::PriorityQeueuType and PriorityQueueAllocExpr
        struct LowerPriorityQueueTypeandAllocExpr : public mir::MIRVisitor {
            using mir::MIRVisitor::visit;

            LowerPriorityQueueTypeandAllocExpr(Schedule* schedule) : schedule_(schedule){

            };

            virtual void visit(mir::PriorityQueueType){

            }

            virtual void visit(mir::PriorityQueueAllocExpr){

            }

            Schedule * schedule_;

        };

        struct LowerUpdatePriorityExternVertexSetApplyExpr : public mir::MIRRewriter {
	    using mir::MIRRewriter::visit;
	    LowerUpdatePriorityExternVertexSetApplyExpr(Schedule *schedule) : schedule_(schedule) {
    
	    }
            virtual void visit(mir::ExprStmt::Ptr);
	     
            Schedule * schedule_;

	};


        void lower();

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;

    };
}


#endif //GRAPHIT_PRIORITY_FEATURES_LOWERING_H
