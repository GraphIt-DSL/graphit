//
// Created by Yunming Zhang on 5/9/17.
//

#include <graphit/midend/mir_lower.h>


namespace graphit {
    void MIRLower::lower(MIRContext* mir_context){
        PhysicalDataLayoutLower(mir_context).lower();
    }
}

