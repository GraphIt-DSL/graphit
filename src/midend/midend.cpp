//
// Created by Yunming Zhang on 2/14/17.
//
#include <graphit/midend/midend.h>
#include <graphit/midend/mir_lower.h>
#include <graphit/midend/mir_context.h>

namespace graphit {
    int Midend::emitMIR(MIRContext* mir_context) {
        MIREmitter(mir_context).emitIR(fir_context_->getProgram());
        MIRLower().lower(mir_context, schedule_);

        return 0;
    }
}