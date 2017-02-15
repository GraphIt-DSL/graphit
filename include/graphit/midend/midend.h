//
// Created by Yunming Zhang on 2/14/17.
//

#ifndef GRAPHIT_MIDEND_H
#define GRAPHIT_MIDEND_H

#include <graphit/frontend/fir_context.h>
#include <graphit/midend/mir_context.h>
#include <graphit/midend/mir_emitter.h>

namespace graphit {
    class Midend {
    public:
        Midend(FIRContext* fir_context) : fir_context_(fir_context) {

        }

        int emitMIR(MIRContext * mir_context);

    private:
        FIRContext* fir_context_;
    };
}
#endif //GRAPHIT_MIDEND_H
