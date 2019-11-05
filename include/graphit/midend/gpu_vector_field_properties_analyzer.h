#ifndef GPU_VECTOR_FIELD_PROPERTIES_ANALYZER_H
#define GPU_VECTOR_FIELD_PROPERTIES_ANALYZER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/field_vector_property.h>
#include <unordered_set>
namespace graphit {

class GPUVectorFieldPropertiesAnalyzer {
	struct PropertyAnalyzingVisitor: public mir::MIRVisitor {
		MIRContext* mir_context_;

		std::unordered_set<std::string> independent_variables;
		mir::FuncDecl::Ptr enclosing_function;

		PropertyAnalyzingVisitor(MIRContext* mir_context, std::unordered_set<std::string> idp, mir::FuncDecl::Ptr ef): mir_context_(mir_context), independent_variables(idp), enclosing_function(ef) {
		}
		
		using mir::MIRVisitor::visit;

		bool is_independent_index(mir::Expr::Ptr);	

		virtual void visit(mir::TensorReadExpr::Ptr) override;	
		virtual void visit(mir::AssignStmt::Ptr) override;
		virtual void visit(mir::ReduceStmt::Ptr) override;
		
		virtual void visit(mir::PriorityUpdateOperatorMin::Ptr) override;
		
	};
	struct ApplyExprVisitor: public mir::MIRVisitor {
		MIRContext* mir_context_;
		ApplyExprVisitor(MIRContext* mir_context): mir_context_(mir_context) {
		}
		using mir::MIRVisitor::visit;
		virtual void visit(mir::PushEdgeSetApplyExpr::Ptr) override;
		virtual void visit(mir::PullEdgeSetApplyExpr::Ptr) override;
		virtual void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr) override;
	};

	MIRContext* mir_context_;
public:
	void analyze(void);
	GPUVectorFieldPropertiesAnalyzer(MIRContext* mir_context, Schedule* schedule): mir_context_(mir_context) {
	}
};

}
#endif

