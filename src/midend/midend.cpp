//
// Created by Yunming Zhang on 2/14/17.
//
#include <graphit/midend/midend.h>
#include <graphit/midend/mir_lower.h>
#include <graphit/midend/mir_context.h>

namespace graphit {
    /**
     * Read in the fir_context and emits the lowest level MIRs and add them to output mir_context
     * @param mir_context
     * @return
     */
    int Midend::emitMIR(MIRContext* mir_context) {
        // MIREmitter first emits high level MIR based on the FIR nodes
        MIREmitter(mir_context).emitIR(fir_context_->getProgram());
        //MIRLower lowers the higher level MIRs into lower level MIRs based on the user specified schedules
        MIRLower().lower(mir_context, schedule_);

        return 0;
    }
}