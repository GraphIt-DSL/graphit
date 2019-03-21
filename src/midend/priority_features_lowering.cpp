//
// Created by Yunming Zhang on 3/20/19.
//

#include <graphit/midend/priority_features_lowering.h>

namespace graphit {

    void PriorityFeaturesLower::lower() {


        auto lower_extern_apply_expr = LowerUpdatePriorityExternVertexSetApplyExpr(schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions) {
            lower_extern_apply_expr.rewrite(function);
        }

    }
    void PriorityFeaturesLower::LowerUpdatePriorityExternVertexSetApplyExpr::visit(mir::ExprStmt::Ptr expr_stmt) {
	std::cout << "Expr stmt\n";
        if (mir::isa<mir::UpdatePriorityExternVertexSetApplyExpr>(expr_stmt->expr)) {
		std::cout << "Found a UpdatePriorityExternVertexSetApplyExpr\n";
	}
	MIRRewriter::visit(expr_stmt);	
    }
}
