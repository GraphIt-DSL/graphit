#ifndef FRONTIER_REUSE_ANALYSIS_H
#define FRONTIER_REUSE_ANALYSIS_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/field_vector_property.h>
#include <unordered_set>
namespace graphit {
class FrontierReuseAnalysis {
public:
	MIRContext *mir_context_;	
	FrontierReuseAnalysis (MIRContext* mir_context): mir_context_(mir_context) {
	}	
	struct ReuseFindingVisitor: public mir::MIRVisitor {
		MIRContext *mir_context_;	
		ReuseFindingVisitor(MIRContext* mir_context): mir_context_(mir_context) {
		}
		using mir::MIRVisitor::visit;	
		std::vector<mir::Stmt::Ptr> to_deletes;
		bool is_frontier_reusable(mir::StmtBlock::Ptr, int index, std::string frontier_name); 
		virtual void visit(mir::StmtBlock::Ptr) override;
	};
	struct FrontierUseFinder: public mir::MIRVisitor {
		using mir::MIRVisitor::visit;
		bool is_used = false;
		std::string frontier_name;

		virtual void visit(mir::VarExpr::Ptr) override;
		virtual void visit(mir::PushEdgeSetApplyExpr::Ptr) override;
		virtual void visit(mir::PullEdgeSetApplyExpr::Ptr) override;
		virtual void visit(mir::EdgeSetApplyExpr::Ptr) override;
		
	};
	void analyze(void);
};

}
#endif
