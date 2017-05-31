//
// Created by Yunming Zhang on 5/9/17.
//

#ifndef GRAPHIT_MIR_LOWER_H
#define GRAPHIT_MIR_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/midend/physical_data_layout_lower.h>
#include <graphit/midend/apply_expr_lower.h>

namespace graphit {
    class MIRLower {
    public:
        MIRLower(){};

        void lower(MIRContext* mir_context, Schedule* schedule);
        
    };
}


#endif //GRAPHIT_MIR_LOWER_H
