//
// Created by Yunming Zhang on 6/29/17.
//

#include <graphit/midend/change_tracking_lower.h>

namespace graphit {

    void ChangeTrackingLower::lower() {
        auto apply_expr_visitor = ApplyExprVisitor(mir_context_, schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions) {
            function->accept(&apply_expr_visitor);
        }
    }

    void ChangeTrackingLower::ApplyExprVisitor::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {
        auto apply_func_decl_name = apply_expr->input_function_name;
        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_func_decl_name);
        std::string tracking_field = apply_expr->tracking_field;
        auto tracking_var_gen_visitor = TrackingVariableGenVisitor(mir_context_);
        apply_func_decl->accept(&tracking_var_gen_visitor);
        if (tracking_field != "") {
            //TODO: another check to see if it is parallel

            insertSerialReturnStmtForTrackingChange(apply_func_decl,
                                                    tracking_field,
                                                    tracking_var_gen_visitor.getFieldTrackingVariableExpr(
                                                            tracking_field));
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
        auto tracking_var_gen_visitor = TrackingVariableGenVisitor(mir_context_);
        apply_func_decl->accept(&tracking_var_gen_visitor);
        if (tracking_field != "") {
            //TODO: another check to see if it is parallel

            insertSerialReturnStmtForTrackingChange(apply_func_decl,
                                                    tracking_field,
                                                    tracking_var_gen_visitor.getFieldTrackingVariableExpr(
                                                            tracking_field));
        }

    }

    /**
     * Inserts return stmt for tracking fields
     */
    void
    ChangeTrackingLower::ApplyExprVisitor::insertSerialReturnStmtForTrackingChange(mir::FuncDecl::Ptr apply_func_decl,
                                                                                   std::string tracking_field,
                                                                                   mir::Expr::Ptr tracking_var) {

        if (apply_func_decl->field_vector_properties_map_.find(tracking_field)
            != apply_func_decl->field_vector_properties_map_.end()) {
            if (apply_func_decl->field_vector_properties_map_[tracking_field].read_write_type
                == FieldVectorProperty::ReadWriteType::WRITE_ONLY) {
                //if the tracking field has been updated, then add the return
                auto bool_type = std::make_shared<mir::ScalarType>();
                bool_type->type = mir::ScalarType::Type::BOOL;
                auto output_var_name = "output" + mir_context_->getUniqueNameCounterString();
                apply_func_decl->result = mir::Var(output_var_name, bool_type);

                auto assign_stmt = std::make_shared<mir::AssignStmt>();
                auto lhs = std::make_shared<mir::VarExpr>();
                lhs->var = mir::Var(output_var_name, bool_type);
                assign_stmt->lhs = lhs;
                assign_stmt->expr = tracking_var;

                apply_func_decl->body->insertStmtEnd(assign_stmt);
            }
        }
    }

    void ChangeTrackingLower::TrackingVariableGenVisitor::visit(mir::AssignStmt::Ptr assign_stmt) {
        //TODO: may be build another visitor for tensor read to figure out the field
        //For now, I am assuming that the left hand side of assign stmt is a tensor read expression
        //It can be a tensor struct read or a tensor array read, but I just need the field name
        if (mir::isa<mir::TensorReadExpr>(assign_stmt->lhs)){
            auto tensor_read_expr = mir::to<mir::TensorReadExpr>(assign_stmt->lhs);
            auto field_vector_target_expr = mir::to<mir::VarExpr>(tensor_read_expr->target);
            auto field_vector_name = field_vector_target_expr->var.getName();
            // it is always true for assign statement
            addFieldTrackingVariable(field_vector_name, "true");
        }
    }

    void ChangeTrackingLower::TrackingVariableGenVisitor::visit(mir::ReduceStmt::Ptr reduce_stmt) {
        //TODO: may be build another visitor for tensor read to figure out the field
        //For now, I am assuming that the left hand side of assign stmt is a tensor read expression
        //It can be a tensor struct read or a tensor array read, but I just need the field name
        auto tensor_read_expr = mir::to<mir::TensorReadExpr>(reduce_stmt->lhs);
        auto field_vector_target_expr = mir::to<mir::VarExpr>(tensor_read_expr->target);
        auto field_vector_name = field_vector_target_expr->var.getName();

        auto field_vector_tracking_var_name = field_vector_name + mir_context_->getUniqueNameCounterString();
        // field vector might not have changed
        //assign the tracking variable name to the assign statement (used for storing the output of CAS)
        addFieldTrackingVariable(field_vector_name, field_vector_tracking_var_name);


    }

    void ChangeTrackingLower::TrackingVariableGenVisitor::addFieldTrackingVariable(std::string field_vector_name,
                                                                                   std::string tracking_var) {
        if (field_vector_variable_map_.find(field_vector_name) == field_vector_variable_map_.end()) {
            field_vector_variable_map_[field_vector_name] = std::vector<std::string>();
        }
        field_vector_variable_map_[field_vector_name].push_back(tracking_var);
    }

    mir::Expr::Ptr
    ChangeTrackingLower::TrackingVariableGenVisitor::getFieldTrackingVariableExpr(std::string field_vector_name) {
        if (field_vector_variable_map_.find(field_vector_name) == field_vector_variable_map_.end()) {
            return nullptr;
        } else if (field_vector_variable_map_[field_vector_name][0] == "true") {
            //for now, assume we only write to the field once
            assert(field_vector_variable_map_[field_vector_name].size() == 1);
            //this is just true
            auto output = std::make_shared<mir::BoolLiteral>();
            output->val = true;
            return output;
        } else {
            //TODO: in the future, if we have multiple writes to the field, may be we need to create a big OR expression

            //for now, assume we only write to the field once
            assert(field_vector_variable_map_[field_vector_name].size() == 1);

            //this is a variable to be set
            auto bool_type = std::make_shared<mir::ScalarType>();
            bool_type->type = mir::ScalarType::Type::BOOL;
            auto output_var_name = field_vector_variable_map_[field_vector_name][0];
            auto output_var = mir::Var(output_var_name, bool_type);
            auto output_var_expr = std::make_shared<mir::VarExpr>();
            output_var_expr->var = output_var;
            return output_var_expr;
        }
    }
}