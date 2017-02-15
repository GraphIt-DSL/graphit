//
// Created by Yunming Zhang on 2/14/17.
//
#include <graphit/midend/midend.h>
#include <graphit/midend/mir_context.h>

namespace graphit {
    int Midend::emitMIR(MIRContext* mir_context) {
        MIREmitter(mir_context).emitIR(fir_context_->getProgram());

        //prints out the MIR
        std::cout << "mir: " << std::endl;
        std::cout << *mir_context->getStatements().front();
        std::cout << std::endl;

        return 0;
    }
}