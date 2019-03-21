//
// Created by Yunming Zhang on 3/20/19.
//

#include <graphit/midend/priority_features_lowering.h>

namespace graphit {

    void PriorityFeaturesLower::lower() {

        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

        auto schedule_finder = PriorityUpdateScheduleFinder(mir_context_, schedule_);
        for (auto function : functions) {
            function->accept(&schedule_finder);
        }

        auto lower_extern_apply_expr = LowerUpdatePriorityExternVertexSetApplyExpr(schedule_);
        for (auto function : functions) {
            lower_extern_apply_expr.rewrite(function);
        }

    }

    void PriorityFeaturesLower::PriorityUpdateScheduleFinder::visit(
            mir::UpdatePriorityEdgeSetApplyExpr::Ptr update_priority_edgeset_apply_expr){
        if (schedule_ != nullptr && schedule_->apply_schedules != nullptr) {
            auto current_label = label_scope_.getCurrentScope();
            setPrioritySchedule(current_label);
        }
    }

    void PriorityFeaturesLower::PriorityUpdateScheduleFinder::visit(
            mir::UpdatePriorityExternVertexSetApplyExpr::Ptr update_priority_extern_vertexset_apply_expr){
        if (schedule_ != nullptr && schedule_->apply_schedules != nullptr) {
            auto current_label = label_scope_.getCurrentScope();
            setPrioritySchedule(current_label);
        }
    }

    void PriorityFeaturesLower::PriorityUpdateScheduleFinder::setPrioritySchedule(std::string current_label){
        auto apply_schedule = schedule_->apply_schedules->find(current_label);
        if (apply_schedule != schedule_->apply_schedules->end()) { //a schedule is found
            if (apply_schedule->second.priority_update_type == ApplySchedule::PriorityUpdateType::REDUCTION_BEFORE_UPDATE){
                mir_context_->priority_update_type = mir::PriorityUpdateType::ReduceBeforePriorityUpdate;
            } else if (apply_schedule->second.priority_update_type == ApplySchedule::PriorityUpdateType::EAGER_PRIORITY_UPDATE){
                mir_context_->priority_update_type = mir::PriorityUpdateType::EagerPriorityUpdate;
            } else if (apply_schedule->second.priority_update_type == ApplySchedule::PriorityUpdateType::CONST_SUM_REDUCTION_BEFORE_UPDATE){
                mir_context_->priority_update_type = mir::PriorityUpdateType::ConstSumReduceBeforePriorityUpdate;
            } else {
                mir_context_->priority_update_type = mir::PriorityUpdateType::EagerPriorityUpdateWithMerge;
            }
        }
    }

    void PriorityFeaturesLower::LowerUpdatePriorityExternVertexSetApplyExpr::visit(mir::ExprStmt::Ptr expr_stmt) {
        if (mir::isa<mir::UpdatePriorityExternVertexSetApplyExpr>(expr_stmt->expr)) {
		std::cout << "Found a UpdatePriorityExternVertexSetApplyExpr\n";
	}
	MIRRewriter::visit(expr_stmt);	
    }
}
