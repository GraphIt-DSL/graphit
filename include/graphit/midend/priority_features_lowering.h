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


        void lower();

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;

    };
}


#endif //GRAPHIT_PRIORITY_FEATURES_LOWERING_H
