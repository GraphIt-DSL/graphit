#include <graphit/midend/frontier_reuse_analysis.h>

namespace graphit {
void FrontierReuseAnalysis::analyze(void) {
	for (auto func: mir_context_->getFunctionList()) {
		ReuseFindingVisitor visitor(mir_context_);
		func->accept(&visitor);	
	}
}
bool FrontierReuseAnalysis::ReuseFindingVisitor::is_frontier_reusable(mir::StmtBlock::Ptr stmt_block, int index, std::string frontier_name) {
	FrontierUseFinder finder;
	finder.frontier_name = frontier_name;
	index++;
	for (int i = index; i < stmt_block->stmts->size(); i++) {
		if (mir::isa<mir::ExprStmt>((*(stmt_block->stmts))[i])) {
			mir::ExprStmt::Ptr expr_stmt = mir::to<mir::ExprStmt>((*(stmt_block->stmts))[i]);
			if (mir::isa<mir::Call>(expr_stmt->expr)) {
				mir::Call::Ptr call_expr = mir::to<mir::Call>(expr_stmt->expr);
				if (call_expr->name == "deleteObject" && mir::isa<mir::VarExpr>(call_expr->args[0]) && mir::to<mir::VarExpr>(call_expr->args[0])->var.getName() == frontier_name) {
					to_deletes.push_back(expr_stmt);
					return true;
				}
			}	
		}	
		(*(stmt_block->stmts))[i]->accept(&finder);
		if (finder.is_used)
			return false;
	}
	return false;
}
void FrontierReuseAnalysis::ReuseFindingVisitor::visit(mir::StmtBlock::Ptr stmt_block) {
	std::vector<mir::Stmt::Ptr> new_stmts;	
	to_deletes.clear();
	for (int i = 0; i < stmt_block->stmts->size(); i++) {
		mir::Stmt::Ptr this_stmt = (*(stmt_block->stmts))[i];
		if (mir::isa<mir::AssignStmt>(this_stmt)) {
			mir::AssignStmt::Ptr assign_stmt = mir::to<mir::AssignStmt>(this_stmt);
			if (mir::isa<mir::EdgeSetApplyExpr>(assign_stmt->expr)) {
				mir::EdgeSetApplyExpr::Ptr esae = mir::to<mir::EdgeSetApplyExpr>(assign_stmt->expr);
				if (esae->from_func && esae->from_func->function_name->name != "" && !mir_context_->isFunction(esae->from_func->function_name->name)) {
					std::string frontier_name = esae->from_func->function_name->name;
					if (is_frontier_reusable(stmt_block, i, frontier_name)) {
						esae->frontier_reusable = true;
					}
				}
			}
		} else if (mir::isa<mir::VarDecl>(this_stmt)) {
			mir::VarDecl::Ptr var_decl = mir::to<mir::VarDecl>(this_stmt);
			if (var_decl->initVal != nullptr) {
				if (mir::isa<mir::EdgeSetApplyExpr>(var_decl->initVal)) {
					mir::EdgeSetApplyExpr::Ptr esae = mir::to<mir::EdgeSetApplyExpr>(var_decl->initVal);
					if (esae->from_func && esae->from_func->function_name->name != "" && !mir_context_->isFunction(esae->from_func->function_name->name)) {
						std::string frontier_name = esae->from_func->function_name->name;
						if (is_frontier_reusable(stmt_block, i, frontier_name)) {
							esae->frontier_reusable = true;
						}
					}
				}	
			}
		}
		if (std::find(to_deletes.begin(), to_deletes.end(), this_stmt) == to_deletes.end()) {
			new_stmts.push_back(this_stmt);	
		}
	}	
	(*(stmt_block->stmts)) = new_stmts;
	mir::MIRVisitor::visit(stmt_block);
}
void FrontierReuseAnalysis::FrontierUseFinder::visit(mir::VarExpr::Ptr var_expr) {
	if (var_expr->var.getName() == frontier_name)
		is_used = true;
}
void FrontierReuseAnalysis::FrontierUseFinder::visit(mir::PushEdgeSetApplyExpr::Ptr pesae) {
	mir::MIRVisitor::visit(pesae);
	if (pesae->from_func->function_name->name == frontier_name)
		is_used = true;
}
void FrontierReuseAnalysis::FrontierUseFinder::visit(mir::PullEdgeSetApplyExpr::Ptr pesae) {
	mir::MIRVisitor::visit(pesae);
	if (pesae->from_func->function_name->name == frontier_name)
		is_used = true;
}
void FrontierReuseAnalysis::FrontierUseFinder::visit(mir::EdgeSetApplyExpr::Ptr esae) {
	mir::MIRVisitor::visit(esae);
	if (esae->from_func->function_name->name == frontier_name)
		is_used = true;
}
}
