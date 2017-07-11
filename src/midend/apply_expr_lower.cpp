//
// Created by Yunming Zhang on 5/30/17.
//

#include <graphit/midend/apply_expr_lower.h>

namespace  graphit {
    //lowers vertexset apply and edgeset apply expressions according to schedules
    void ApplyExprLower::lower() {
        auto lower_apply_expr = LowerApplyExpr(schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions){
            lower_apply_expr.rewrite(function);
        }
    }


    void ApplyExprLower::LowerApplyExpr::visit(mir::EdgeSetApplyExpr::Ptr edgeset_apply) {
        // check if the schedule contains entry for the current edgeset apply expressions

        if(schedule_ !=nullptr && schedule_->apply_schedules != nullptr) {
            // We assume that there is only one apply in each statement
            auto current_scope_name = label_scope_.getCurrentScope();
            auto apply_schedule = schedule_->apply_schedules->find(current_scope_name);

            if (apply_schedule != schedule_->apply_schedules->end()) {
                // a schedule is found

                if (apply_schedule->second.direction_type == ApplySchedule::DirectionType::PUSH) {
                    node = std::make_shared<mir::PushEdgeSetApplyExpr>(edgeset_apply);
                    return;
                } else if (apply_schedule->second.direction_type == ApplySchedule::DirectionType::PULL){
                    //Pull
                    node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
                    return;
                } else if  (apply_schedule->second.direction_type == ApplySchedule::DirectionType::HYBRID){
                    //Hybrid
                    //TODO: not yet supported
                }


                if (apply_schedule->second.parallel_type == ApplySchedule::ParType::Parallel){
                    mir::to<mir::EdgeSetApplyExpr>(node)->is_parallel = true;
                } else if (apply_schedule->second.parallel_type == ApplySchedule::ParType::Serial){
                    mir::to<mir::EdgeSetApplyExpr>(node)->is_parallel = false;
                }

                if (apply_schedule->second.deduplication_type == ApplySchedule::DeduplicationType::Enable){
                    mir::to<mir::EdgeSetApplyExpr>(node)->enable_deduplication = true;
                } else if (apply_schedule->second.deduplication_type == ApplySchedule::DeduplicationType ::Disable){
                    mir::to<mir::EdgeSetApplyExpr>(node)->enable_deduplication = false;
                }

            } else {
                //There is a schedule, but nothing is specified for the current apply
                node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
                return;
            }


            return;
        }else {
            node = std::make_shared<mir::PullEdgeSetApplyExpr>(edgeset_apply);
            return;
        }
    }
}