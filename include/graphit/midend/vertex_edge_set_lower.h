//
// Created by Yunming Zhang on 7/24/17.
//

#ifndef GRAPHIT_VERTEX_EDGE_SET_LOWER_H
#define GRAPHIT_VERTEX_EDGE_SET_LOWER_H

#include <graphit/midend/mir_context.h>


namespace graphit {

/**
 * Lowers declarations of vertexset and edgeset into relevant main function statements
 */
    class VertexEdgeSetLower {
    public:
        VertexEdgeSetLower(MIRContext* mir_context) : mir_context_(mir_context){}

        void lower();
    private:
        MIRContext *mir_context_;
    };
}

#endif //GRAPHIT_VERTEX_EDGE_SET_LOWER_H
