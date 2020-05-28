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

        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_expr->input_function->function_name->name);
        auto edgeset_str = mir::to<mir::VarExpr>(apply_expr->target)->var.getName();
        auto merge_reduce = std::make_shared<mir::MergeReduceField>();
        mir_context_->edgeset_to_label_to_merge_reduce[edgeset_str][apply_schedule->second.scope_label_name] = merge_reduce;
        if (apply_schedule->second.numa_aware) {
            auto int_type = std::make_shared<mir::ScalarType>();
            int_type->type = mir::ScalarType::Type::INT;
            apply_func_decl->args.push_back(mir::Var("socketId", int_type));
            merge_reduce->numa_aware = true;
        }


        auto reduce_stmt_visitor = ReduceStmtVisitor(mir_context_, merge_reduce);
        apply_func_decl->accept(&reduce_stmt_visitor);
        apply_expr->merge_reduce = merge_reduce;
    }

    void MergeReduceLower::ReduceStmtVisitor::visit(mir::ReduceStmt::Ptr reduce_stmt) {
        if (mir::isa<mir::TensorReadExpr>(reduce_stmt->lhs)) {
            auto tensor_read_expr = mir::to<mir::TensorReadExpr>(reduce_stmt->lhs);
            if (!mir::isa<mir::VarExpr>(tensor_read_expr->target))
                return;
            auto target_expr = mir::to<mir::VarExpr>(tensor_read_expr->target);
            merge_reduce_->field_name = target_expr->var.getName();

            // TODO this is a hack to get around with local vector initialization. getVectorItemType only works for global arrays.
            // But it works fine for now, because this optimization won't work for local arrays.
            if (mir_context_->getVectorItemType(merge_reduce_->field_name) != nullptr) {
                merge_reduce_->scalar_type = mir::to<mir::ScalarType>(mir_context_->getVectorItemType(merge_reduce_->field_name));
            }

            merge_reduce_->reduce_op = reduce_stmt->reduce_op_;

            if (merge_reduce_->numa_aware) {
                target_expr->var = mir::Var("local_" + merge_reduce_->field_name + "[socketId]", target_expr->var.getType());
            }
        }
    }
}
