//
// Created by Yunming Zhang on 6/22/17.
//

#ifndef GRAPHIT_VECTOR_OP_LOWER_H
#define GRAPHIT_VECTOR_OP_LOWER_H

#include <graphit/midend/mir_context.h>

namespace graphit {
    /**
     * Lowers operations on vectors into apply operations
     * Operations include, const / var decl with loaded values
     * TODO: Element wise +, * ..
     */
    class VectorOpLower {
    public:
        VectorOpLower(MIRContext *mir_context) : mir_context_(mir_context){

        }

        //lowers the global constant vector declarations
        void lowerConstVectorVarDecl();

        //TODO: lowers other vector ops
        void lower();

    private:
        MIRContext *mir_context_;

    };
}
#endif //GRAPHIT_VECTOR_OP_LOWER_H
