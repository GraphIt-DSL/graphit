#include "graphit/backend/codegen_gpu/extract_read_write_set.h"

namespace graphit {
void ExtractReadWriteSet::visit(mir::StmtBlock::Ptr stmt_block) {
	return;
}
void ExtractReadWriteSet::visit(mir::TensorArrayReadExpr::Ptr tare) {
	mir::Var target = mir::to<mir::VarExpr>(tare->target)->var;
	if (mir_context->isLoweredConstTensor(target.getName())) {
		add_read(tare);
	}
	tare->index->accept(this);
}
void ExtractReadWriteSet::visit(mir::AssignStmt::Ptr assign_stmt) {
	if (mir::isa<mir::TensorArrayReadExpr>(assign_stmt->lhs)) {
		mir::TensorArrayReadExpr::Ptr tare = mir::to<mir::TensorArrayReadExpr>(assign_stmt->lhs);
		mir::Var target = mir::to<mir::VarExpr>(tare->target)->var;
		if (mir_context->isLoweredConstTensor(target.getName())) {
			add_write(tare);
		tare->index->accept(this);
		assign_stmt->expr->accept(this);
	}
	tare->index->accept(this);
		
	} else {
		assign_stmt->lhs->accept(this);
		assign_stmt->expr->accept(this);
	}
}
void ExtractReadWriteSet::add_read(mir::TensorArrayReadExpr::Ptr tare) {
	read_set_.push_back(tare);
}
void ExtractReadWriteSet::add_write(mir::TensorArrayReadExpr::Ptr tare) {
	write_set_.push_back(tare);
}
}
