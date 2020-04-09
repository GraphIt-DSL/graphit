//
// Created by Tugsbayasgalan Manlaibaatar on 2020-04-08.
//

#ifndef GRAPHIT_PAR_FOR_LOWER_H
#define GRAPHIT_PAR_FOR_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {
    class ParForLower {
    public:
        // construct with no input schedule
        ParForLower(MIRContext *mir_context) : mir_context_(mir_context) {

        }

        //constructor with input schedule
        ParForLower(MIRContext *mir_context, Schedule *schedule)
                : schedule_(schedule), mir_context_(mir_context) {};


        void lower();

        struct LowerParForStmt : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerParForStmt(Schedule* schedule, MIRContext* mir_context)
                    : schedule_(schedule), mir_context_(mir_context){

            };

            virtual void visit(mir::ParForStmt::Ptr par_for);

            Schedule * schedule_;
            MIRContext* mir_context_;
        };


    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;
    };
}



#endif //GRAPHIT_PAR_FOR_LOWER_H
