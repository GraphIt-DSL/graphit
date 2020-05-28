//
// Created by Tugsbayasgalan Manlaibaatar on 2019-10-08.
//

#include <graphit/midend/intersection_expr_lower.h>

namespace graphit {

    void IntersectionExprLower::lower() {
        auto lower_intersection_expr = LowerIntersectionExpr(schedule_, mir_context_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions) {
            lower_intersection_expr.rewrite(function);
        }

    }

    void IntersectionExprLower::LowerIntersectionExpr::visit(mir::IntersectionExpr::Ptr intersection_expr) {
        // by default the intersection type should be naive
        intersection_expr->intersectionType = IntersectionSchedule::IntersectionType::NAIVE;
        if (schedule_ != nullptr && schedule_->intersection_schedules != nullptr) {
            // We assume that there is only one intersect in each statement
            auto current_scope_name = label_scope_.getCurrentScope();
            auto intersection_schedule = schedule_->intersection_schedules->find(current_scope_name);
            if (intersection_schedule != schedule_->intersection_schedules->end()) {
                //if a schedule for the statement has been found
                intersection_expr->intersectionType = intersection_schedule->second;
            }
        }
        node = intersection_expr;

    }

    void IntersectionExprLower::LowerIntersectionExpr::visit(mir::IntersectNeighborExpr::Ptr intersection_expr) {
        // by default the intersection type should be naive
        intersection_expr->intersectionType = IntersectionSchedule::IntersectionType::NAIVE;
        if (schedule_ != nullptr && schedule_->intersection_schedules != nullptr) {
            // We assume that there is only one intersect in each statement
            auto current_scope_name = label_scope_.getCurrentScope();
            auto intersection_schedule = schedule_->intersection_schedules->find(current_scope_name);
            if (intersection_schedule != schedule_->intersection_schedules->end()) {
                //if a schedule for the statement has been found
                intersection_expr->intersectionType = intersection_schedule->second;
            }
        }
        node = intersection_expr;

    }

}