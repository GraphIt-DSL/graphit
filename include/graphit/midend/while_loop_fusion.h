#ifndef WHILE_LOOP_FUSION_H
#define WHILE_LOOP_FUSION_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {

struct WhileLoopFusion: public mir::MIRVisitor {
	using mir::MIRVisitor::visit;
	WhileLoopFusion(MIRContext* mir_context, Schedule* schedule): mir_context_(mir_context), schedule_(schedule) {
	}
	void lower(void);
protected:
	virtual void visit(mir::WhileStmt::Ptr);
private:
	Schedule *schedule_ = nullptr;
	MIRContext *mir_context_ = nullptr;
};

}

#endif
