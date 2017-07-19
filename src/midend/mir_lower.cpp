//
// Created by Yunming Zhang on 5/9/17.
//

#include <graphit/midend/mir_lower.h>
#include <graphit/midend/physical_data_layout_lower.h>
#include <graphit/midend/apply_expr_lower.h>
#include <graphit/midend/vector_op_lower.h>
#include <graphit/midend/change_tracking_lower.h>
#include <graphit/midend/vector_field_properties_analyzer.h>
#include <graphit/midend/atomics_op_lower.h>

namespace graphit {
    /**
     * Perfomrms the lowering passes on MIR_Context
     * @param mir_context
     * @param schedule
     */
    void MIRLower::lower(MIRContext* mir_context, Schedule* schedule){
        VectorOpLower(mir_context).lower();
        // This pass lowers apply expression into concrete versions, including push, pull hybrid and more
        ApplyExprLower(mir_context, schedule).lower();

        // Use program analysis to figure out the read/write/read and write local / shared properties of the fields
        VectorFieldPropertiesAnalyzer(mir_context,schedule).analyze();

        // The pass on lowering abstract data structures to
        // concrete data structures with physical layout information
        PhysicalDataLayoutLower(mir_context, schedule).lower();

        // This pass inserts atomic operations including CAS, writeMin, writeAdd
        // This pass does not need the schedule
        //AtomicsOpLower(mir_context).lower();

        // This pass generates return values for implicit tracking of changes to certain field
        ChangeTrackingLower(mir_context, schedule).lower();



    }
}

