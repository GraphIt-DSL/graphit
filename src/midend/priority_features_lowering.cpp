//
// Created by Yunming Zhang on 3/20/19.
//

#include <graphit/midend/priority_features_lowering.h>

namespace graphit {

    void PriorityFeaturesLower::lower() {


        auto lower_extern_apply_expr = LowerUpdatePriorityExternVertexSetApplyExpr(schedule_, mir_context_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions) {
            lower_extern_apply_expr.rewrite(function);
        }

    }
    void PriorityFeaturesLower::LowerUpdatePriorityExternVertexSetApplyExpr::visit(mir::ExprStmt::Ptr expr_stmt) {
	MIRRewriter::visit(expr_stmt);	
        if (mir::isa<mir::UpdatePriorityExternVertexSetApplyExpr>(expr_stmt->expr)) {

/*
		mir::UpdatePriorityExternCall::Ptr call_stmt = make_shared<mir::UpdatePriorityExternCall>();
		call_stmt->input_set = expr_stmt->target;
		call_stmt->apply_function_name = expr_stmt->apply_function_name;
		call_stmt->lambda_name = mir_context_->getUniqueNameCounterString();
		call_stmt->output_set_name = mir_context_->getUniqueNameCounterString();	
		
			
		mir::UpdatePriorityUpdateBucketsCall::Ptr update_call = make_shared<mir::UpdatePriorityUpdateBucketsCall>();
		update_call->lambda_name = call_stmt->lambda_name;
		call_stmt->modified_vertexsubset_name = call_stmt->output_set_name;
		
		mir::StmtBlock::Ptr stmt_block = make_shared<mir::StmtBlock>();
*/
	}
	node = expr_stmt;
    }
}
