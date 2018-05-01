//
// Created by Yunming Zhang on 6/22/17.
//

#ifndef GRAPHIT_VECTOR_OP_LOWER_H
#define GRAPHIT_VECTOR_OP_LOWER_H

#include <graphit/midend/mir_context.h>

namespace graphit {
    /**
     * Lowers field vector declartions into global variable declaration
     * Main function declaration and main function vertex set apply expressions (for initialization)
     * The vertexset apply expressions is good for transforming between arrays and structs (with tensor read lowering)
     */
    class GlobalFieldVectorLower {
    public:
        GlobalFieldVectorLower(MIRContext *mir_context) : mir_context_(mir_context){

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
