//
// Created by Yunming Zhang on 2/15/17.
//

#ifndef GRAPHIT_FIR_CONTEXT_H
#define GRAPHIT_FIR_CONTEXT_H

#include <graphit/frontend/fir.h>

namespace graphit {

    class FIRContext {
    public:
        FIRContext() {
        }


        ~FIRContext() {
        }

        void setProgram(fir::Program::Ptr program){
            fir_program_  = program;
        }

        fir::Program::Ptr getProgram(){
            return fir_program_;
        }

    private:
        fir::Program::Ptr fir_program_;
    };
}


#endif //GRAPHIT_FIR_CONTEXT_H
