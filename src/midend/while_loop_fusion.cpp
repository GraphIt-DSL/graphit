#include <graphit/midend/while_loop_fusion.h>
#include <graphit/frontend/abstract_schedule.h>
#include <graphit/frontend/gpu_schedule.h>

void graphit::WhileLoopFusion::lower(void) {	
    std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
    for (auto function : functions) {
        function->accept(this);
    }
}
void graphit::WhileLoopFusion::visit(mir::WhileStmt::Ptr while_stmt) {
    while_stmt->setMetadata<bool>("is_fused", false);
	if (while_stmt->stmt_label != "") {
		label_scope_.scope(while_stmt->stmt_label);
	}
	while_stmt->cond->accept(this);
	while_stmt->body->accept(this);
	if (schedule_->backend_identifier == Schedule::BackendID::GPU) {
      if (while_stmt->hasApplySchedule()) {
        auto apply_schedule = while_stmt->getApplySchedule();
        if (!apply_schedule->isComposite()) {
            auto applied_simple_schedule = apply_schedule->self<fir::gpu_schedule::SimpleGPUSchedule>();
            if (applied_simple_schedule->kernel_fusion == fir::gpu_schedule::SimpleGPUSchedule::kernel_fusion_type::FUSION_ENABLED) {
                while_stmt->setMetadata<bool>("is_fused", true);
                mir_context_->fused_while_loops.push_back(while_stmt);
            }
        }
      }
	}

	if (while_stmt->stmt_label != "") {
		label_scope_.unscope();
	}

}
