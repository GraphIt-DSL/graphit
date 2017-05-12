//
// Created by Yunming Zhang on 2/14/17.
//

#ifndef GRAPHIT_MIDEND_H
#define GRAPHIT_MIDEND_H

#include <graphit/frontend/fir_context.h>
#include <graphit/midend/mir_context.h>
#include <graphit/midend/mir_emitter.h>
#include <graphit/frontend/schedule.h>

namespace graphit {
    class Midend {
    public:
        Midend(FIRContext* fir_context) : fir_context_(fir_context) {

        }

        Midend(FIRContext* fir_context, Schedule * schedule)
                : fir_context_(fir_context), schedule_(schedule) {

        }

        int emitMIR(MIRContext * mir_context);

    private:
        Schedule* schedule_ = nullptr;
        FIRContext* fir_context_ = nullptr;
    };
}
#endif //GRAPHIT_MIDEND_H
