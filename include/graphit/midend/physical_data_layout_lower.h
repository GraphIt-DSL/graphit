//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_LOWERPHYSICALDATALAYOUT_H
#define GRAPHIT_LOWERPHYSICALDATALAYOUT_H

#include <graphit/midend/mir_context.h>

namespace graphit {

    class PhysicalDataLayoutLower {
    public:

        PhysicalDataLayoutLower(){};

        void lower(MIRContext * mir_context);

    };


}

#endif //GRAPHIT_LOWERPHYSICALDATALAYOUT_H
