//
// Created by Sherry Yang on 2/27/18.
//

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/merge_reduce_lower.h>

namespace graphit {

    void MergeReduceLower::lower() {
        auto apply_expr_visitor = ApplyExprVisitor(mir_context_, schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions) {
            function->accept(&apply_expr_visitor);
        }
    }

    void MergeReduceLower::ApplyExprVisitor::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {
        processMergeReduce(apply_expr);
    }

    void MergeReduceLower::ApplyExprVisitor::visit(mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr) {
        processMergeReduce(apply_expr);
    }

    void MergeReduceLower::ApplyExprVisitor::processMergeReduce(mir::EdgeSetApplyExpr::Ptr apply_expr) {
        if (schedule_ == nullptr || schedule_->apply_schedules == nullptr) {
            return;
        };

        // We assume that there is only one apply in each statement
        auto current_scope_name = label_scope_.getCurrentScope();
        auto apply_schedule = schedule_->apply_schedules->find(current_scope_name);
        if (apply_schedule == schedule_->apply_schedules->end()) {
            return;
        }

        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_expr->input_function_name);
        if (apply_schedule->second.numa_aware) {
            auto int_type = std::make_shared<mir::ScalarType>();
            int_type->type = mir::ScalarType::Type::INT;
            apply_func_decl->args.push_back(mir::Var("socketId", int_type));
            mir_context_->numa_aware = true;
        }
        auto reduce_stmt_visitor = ReduceStmtVisitor(mir_context_);
        apply_func_decl->accept(&reduce_stmt_visitor);
        apply_expr->merge_field = reduce_stmt_visitor.merge_field;
        apply_expr->reduce_op = reduce_stmt_visitor.reduce_op;
    }

    void MergeReduceLower::ReduceStmtVisitor::visit(mir::ReduceStmt::Ptr reduce_stmt) {
        if (mir::isa<mir::TensorReadExpr>(reduce_stmt->lhs)) {
            auto tensor_read_expr = mir::to<mir::TensorReadExpr>(reduce_stmt->lhs);
            auto target_expr = mir::to<mir::VarExpr>(tensor_read_expr->target);
            merge_field = target_expr->var.getName();
            reduce_op = reduce_stmt->reduce_op_;
            if (mir_context_->numa_aware) {
                target_expr->var = mir::Var("local_" + merge_field + "[socketId]", target_expr->var.getType());
            }

            auto init_stmt = std::make_shared<mir::LocalFieldInitStmt>();
            init_stmt->merge_field = merge_field;
            init_stmt->scalar_type = mir::to<mir::ScalarType>(mir_context_->getVectorItemType(merge_field));
            mir_context_->local_field_init_stmts.push_back(init_stmt);
        }
    }
}