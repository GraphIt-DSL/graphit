#ifndef GPU_CHANGE_TRACKING_LOWER_H
#define GPU_CHANGE_TRACKING_LOWER_H

#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
namespace graphit {
class GPUChangeTrackingLower {
public:
	MIRContext *mir_context_;
	Schedule *schedule_;
	GPUChangeTrackingLower(MIRContext *mir_context, Schedule *schedule): mir_context_(mir_context), schedule_(schedule) {
	}
	void lower (void);
	struct UdfArgChangeVisitor: public mir::MIRVisitor {
		using mir::MIRVisitor::visit;
		MIRContext *mir_context_;
		UdfArgChangeVisitor(MIRContext *mir_context): mir_context_(mir_context) {
		}
		void updateUdf(mir::FuncDecl::Ptr func_decl, mir::EdgeSetApplyExpr::Ptr);
		virtual void visit(mir::PushEdgeSetApplyExpr::Ptr) override;
		virtual void visit(mir::PullEdgeSetApplyExpr::Ptr) override;
		virtual void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr) override;
	};

	struct ReductionOpChangeVisitor: public mir::MIRVisitor {
		using mir::MIRVisitor::visit;
		MIRContext *mir_context_;
		mir::EdgeSetApplyExpr::Ptr current_edge_set_apply_expr;
		std::string udf_tracking_var;
		mir::Type::Ptr frontier_type;
		ReductionOpChangeVisitor(MIRContext *mir_context, std::string tracking_var, mir::EdgeSetApplyExpr::Ptr edge_set_apply_expr, mir::Type::Ptr type): mir_context_(mir_context), udf_tracking_var(tracking_var), current_edge_set_apply_expr(edge_set_apply_expr), frontier_type(type) {
		}	
		virtual void visit(mir::StmtBlock::Ptr) override;


	};
};
}

#endif
