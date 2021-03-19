//
// Created by Yunming Zhang on 5/9/17.
//

#include <graphit/midend/mir_lower.h>
#include <graphit/midend/physical_data_layout_lower.h>
#include <graphit/midend/apply_expr_lower.h>
#include <graphit/midend/intersection_expr_lower.h>
#include <graphit/midend/par_for_lower.h>
#include <graphit/midend/vector_op_lower.h>
#include <graphit/midend/change_tracking_lower.h>
#include <graphit/midend/gpu_change_tracking_lower.h>
#include <graphit/midend/vector_field_properties_analyzer.h>
#include <graphit/midend/gpu_vector_field_properties_analyzer.h>
#include <graphit/midend/atomics_op_lower.h>
#include <graphit/midend/vertex_edge_set_lower.h>
#include <graphit/midend/merge_reduce_lower.h>
#include <graphit/midend/udf_dup.h>
#include <graphit/midend/priority_features_lowering.h>
#include <graphit/midend/gpu_priority_features_lowering.h>
#include <graphit/midend/while_loop_fusion.h>
#include <graphit/midend/frontier_reuse_analysis.h>

namespace graphit {
    /**
     * Perfomrms the lowering passes on MIR_Context
     * @param mir_context
     * @param schedule
     */
    void MIRLower::lower(MIRContext* mir_context, Schedule* schedule){
    
        // Duplicate the UDFs that have been used in two different EdgeSetApplyExpr
        // before performing any analysis
        UDFReuseFinder(mir_context).lower();

        //lower global vector assignment to vector operations
        GlobalFieldVectorLower(mir_context, schedule).lower();

        //lower  global edgeset assignment (from loading)
        // needed for reading commandline arguments in the main function
        VertexEdgeSetLower(mir_context).lower();


	// We use the GPU version when the GPU Scheules are set
	if (schedule != nullptr && !schedule->apply_gpu_schedules.empty()) {
		GPUPriorityFeaturesLowering(mir_context, schedule).lower();
	} else  {
		//This pass needs to happen before ApplyExprLower pass because the default ReduceBeforeUpdate uses ApplyExprLower
		PriorityFeaturesLower(mir_context, schedule).lower();
	}

	// This pass finds EdgeSetApplyExpressions that allow frontiers to be reused and removes the corresponding deletes
	FrontierReuseAnalysis(mir_context).analyze();

        // This pass sets properties of edgeset apply expressions based on the schedules including
        // edge traversal direction: push, pull, denseforward, hybrid_dense, hybrid_denseforward
        // deduplication: enable / disable
        // parallelization: enable / disable
        // frontier data structure: regular / sliding queue
        // This pass usually needs to be executed earlier to specialize edgeset apply operators,
        //  sets the flags for other parts of the lowering process
        ApplyExprLower(mir_context, schedule).lower();

        // This pass sets properties of intersection operations based on scheduling languages.
        // intersection types: HiroshiIntersection, Naive, Multiskip, Binary, Combined
        // If there is no schedule specified, it just chooses naive intersection.
        IntersectionExprLower(mir_context, schedule).lower();

        // This pass sets grain size of the parallel for. If nothing is given, it will use default OPENMP for loop.
        ParForLower(mir_context, schedule).lower();

        // Use program analysis to figure out the properties of each tensor access
        // read write type: read/write/read and write (reduction)
        // access type: shared or local
	if (schedule != nullptr && !schedule->apply_gpu_schedules.empty()) {
		GPUVectorFieldPropertiesAnalyzer(mir_context, schedule).analyze();
	} else {
		VectorFieldPropertiesAnalyzer(mir_context, schedule).analyze();
	}

        // The pass on lowering abstract data structures to
        // concrete data structures with physical layout information (arrays, field of a struct, dictionary)
        PhysicalDataLayoutLower(mir_context, schedule).lower();

        // This pass inserts atomic operations, including CAS, writeMin, writeAdd
        // This pass does not need the schedule
        AtomicsOpLower(mir_context, schedule).lower();

        // This pass generates code for tracking if a field has been modified
        // during the execution of the edgeset apply functions.
        // It return values for implicit tracking of changes to certain field
	if (schedule != nullptr && !schedule->apply_gpu_schedules.empty()) {
		// No change tracking lower for GPUs
		GPUChangeTrackingLower(mir_context, schedule).lower();
	} else {	
        	ChangeTrackingLower(mir_context, schedule).lower();
	}

        // This pass extracts the merge field and reduce operator. If numa_aware is set to true in
        // the schedule for the corresponding label, it also adds NUMA optimization
        MergeReduceLower(mir_context, schedule).lower();

	// This pass lowers while loops that have fusion schedule attached to them 
	WhileLoopFusion(mir_context, schedule).lower();	

    }
}

