#include <graphit/midend/while_loop_fusion.h>

void graphit::WhileLoopFusion::lower(void) {	
    std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
    for (auto function : functions) {
        function->accept(this);
    }
}
void graphit::WhileLoopFusion::visit(mir::WhileStmt::Ptr while_stmt) {
	if (while_stmt->stmt_label != "") {
		label_scope_.scope(while_stmt->stmt_label);
	}
	while_stmt->cond->accept(this);
	while_stmt->body->accept(this);
	if (schedule_ != nullptr && !schedule_->apply_gpu_schedules.empty()) {
		auto current_scope_name = label_scope_.getCurrentScope();
		auto apply_schedule_iter = schedule_->apply_gpu_schedules.find(current_scope_name);
		if (apply_schedule_iter != schedule_->apply_gpu_schedules.end()) {
			auto apply_schedule = apply_schedule_iter->second;
			if (dynamic_cast<fir::gpu_schedule::SimpleGPUSchedule*>(apply_schedule)) {
				auto applied_simple_schedule = dynamic_cast<fir::gpu_schedule::SimpleGPUSchedule*>(apply_schedule);
				if (applied_simple_schedule->kernel_fusion == fir::gpu_schedule::SimpleGPUSchedule::kernel_fusion_type::FUSION_ENABLED) {
					while_stmt->is_fused = true; 
					mir_context_->fused_while_loops.push_back(while_stmt);
				}
			}
		}
	}

	if (while_stmt->stmt_label != "") {
		label_scope_.unscope();
	}

}

