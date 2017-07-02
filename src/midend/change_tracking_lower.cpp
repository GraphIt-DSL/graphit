//
// Created by Yunming Zhang on 6/29/17.
//

#include <graphit/midend/change_tracking_lower.h>

namespace graphit {

    void ChangeTrackingLower::lower() {
        auto apply_expr_visitor = ApplyExprVisitor(mir_context_, schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions){
            function->accept(&apply_expr_visitor);
        }
    }

    void ChangeTrackingLower::ApplyExprVisitor::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {
        auto apply_func_decl_name = apply_expr->input_function_name;
        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_func_decl_name);
        std::string tracking_field = apply_expr->tracking_field;
        if (tracking_field != "") {
            insertSerialReturnStmtForTrackingChange(apply_func_decl, tracking_field);
        }
    }

    /**
         * Inserts return stmt for tracking fields
         */
    void insertSerialReturnStmtForTrackingChange(mir::FuncDecl apply_func_decl, std::string tracking_field);

    void ChangeTrackingLower::ApplyExprVisitor::visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr) {
        auto apply_func_decl_name = apply_expr->input_function_name;
        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_func_decl_name);
        std::string tracking_field = apply_expr->tracking_field;
        if (tracking_field != "") {
            //TODO: another check to see if it is parallel
            insertSerialReturnStmtForTrackingChange(apply_func_decl, tracking_field);
        }

    }

    /**
     * Inserts return stmt for tracking fields
     */
    void
    ChangeTrackingLower::ApplyExprVisitor::insertSerialReturnStmtForTrackingChange(mir::FuncDecl::Ptr apply_func_decl,
                                                                                   std::string tracking_field) {

        if (apply_func_decl->field_vector_properties_map_.find(tracking_field)
            != apply_func_decl->field_vector_properties_map_.end()) {
            if (apply_func_decl->field_vector_properties_map_[tracking_field].read_write_type
                == FieldVectorProperty::ReadWriteType::WRITE_ONLY) {
                //if the tracking field has been updated, then add the return
                auto bool_type = std::make_shared<mir::ScalarType>();
                bool_type->type = mir::ScalarType::Type ::BOOL;
                auto output_var_name = "output" + mir_context_->getUniqueNameCounterString();
                apply_func_decl->result = mir::Var(output_var_name, bool_type);

                auto assign_stmt = std::make_shared<mir::AssignStmt>();
                auto lhs = std::make_shared<mir::VarExpr>();
                lhs->var = mir::Var(output_var_name, bool_type);
                auto rhs = std::make_shared<mir::BoolLiteral>();
                rhs->val = true;
                assign_stmt->lhs = lhs;
                assign_stmt->expr = rhs;

                apply_func_decl->body->insertStmtEnd(assign_stmt);
            }
        }
    }
}