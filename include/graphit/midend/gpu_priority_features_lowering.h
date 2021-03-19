#ifndef GPU_PRIORITY_FEATURES_LOWERING_H
#define GPU_PRIORITY_FEATURES_LOWERING_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {
class GPUPriorityFeaturesLowering {
public:
	MIRContext *mir_context_;
	Schedule *schedule_;
	GPUPriorityFeaturesLowering(MIRContext* mir_context, Schedule* schedule): mir_context_(mir_context), schedule_(schedule) {
	}
	void lower(void);		


	struct EdgeSetApplyPriorityRewriter: public mir::MIRRewriter {
		MIRContext *mir_context_;
		Schedule *schedule_;
		EdgeSetApplyPriorityRewriter(MIRContext* mir_context, Schedule* schedule): mir_context_(mir_context), schedule_(schedule) {
		}
		
		using mir::MIRRewriter::visit;
		virtual void visit(mir::ExprStmt::Ptr) override;
		
	};
	struct PriorityUpdateOperatorRewriter: public mir::MIRRewriter {
		MIRContext *mir_context_;
		mir::UpdatePriorityEdgeSetApplyExpr::Ptr puesae_;
		PriorityUpdateOperatorRewriter(MIRContext* mir_context, mir::UpdatePriorityEdgeSetApplyExpr::Ptr puesae): mir_context_(mir_context), puesae_(puesae) {
		}
		using mir::MIRRewriter::visit;
		virtual void visit(mir::Call::Ptr) override;
		
	};
	struct UDFPriorityQueueFinder: public mir::MIRVisitor {
		using mir::MIRVisitor::visit;
		
		MIRContext *mir_context_;
		UDFPriorityQueueFinder(MIRContext* mir_context): mir_context_(mir_context) {
		}
		std::vector<mir::Var> priority_queues_used;
		mir::Var getPriorityQueue(void);
		void insertVar(mir::Var);
		virtual void visit(mir::PriorityUpdateOperator::Ptr) override;
		virtual void visit(mir::PriorityUpdateOperatorMin::Ptr) override;
		virtual void visit(mir::PriorityUpdateOperatorSum::Ptr) override;
		virtual void visit(mir::Call::Ptr) override;
	};
};
}

#endif



