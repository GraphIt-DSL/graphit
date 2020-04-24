//
// Created by Yunming Zhang on 6/29/17.
//

#include <graphit/midend/vector_field_properties_analyzer.h>

namespace graphit {

    void VectorFieldPropertiesAnalyzer::analyze() {
        auto apply_expr_visitor = ApplyExprVisitor(mir_context_, schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions) {
            function->accept(&apply_expr_visitor);
        }
    }

    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor
    ::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {
        analyzeSingleFunctionEdgesetApplyExpr(apply_expr->input_function->function_name->name, "pull");
    }

    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor
    ::visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr) {
        analyzeSingleFunctionEdgesetApplyExpr(apply_expr->input_function->function_name->name, "push");
    }

    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor
    ::visit(mir::HybridDenseForwardEdgeSetApplyExpr::Ptr apply_expr) {
        analyzeSingleFunctionEdgesetApplyExpr(apply_expr->input_function->function_name->name, "push");
    }

    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor::visit(mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr) {
        analyzeSingleFunctionEdgesetApplyExpr(apply_expr->input_function->function_name->name, "pull");
        analyzeSingleFunctionEdgesetApplyExpr(apply_expr->push_function_->function_name->name, "push");
    }

    // Analyze the read / write properties for the priority update edgeset apply function
    // For now, only push direction is supported for EagerUpdates
    // TODO figure out how this would interact with the original edgeset apply
    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor::visit(
            mir::UpdatePriorityEdgeSetApplyExpr::Ptr priority_update_expr) {
        if (mir_context_->priority_update_type == mir::EagerPriorityUpdate ||
                mir_context_->priority_update_type == mir::EagerPriorityUpdateWithMerge){
            analyzeSingleFunctionEdgesetApplyExpr(priority_update_expr->input_function->function_name->name, "push");
        } else {

        }
    }

    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor::analyzeSingleFunctionEdgesetApplyExpr(
            std::string function_name, std::string direction) {

        // The analysis only makes sense if it is a parallel apply expr
        auto property_visitor = PropertyAnalyzingVisitor(direction, mir_context_);
        auto apply_func_decl_name = function_name;

        if (!mir_context_->isExternFunction(apply_func_decl_name)){
            //Only perform the property analysis for non-extern function
            mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_func_decl_name);
            apply_func_decl->accept(&property_visitor);
        }
    }

    void VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::PriorityUpdateOperatorSum::Ptr op) {
        if (direction_ == "push") {
            op->is_atomic = true;
        } else {
            op->is_atomic = false;
        }
        enclosing_func_decl_->field_vector_properties_map_[mir_context_->getPriorityVectorName()] = buildLocalReadWriteFieldProperty();
    }

    void VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::PriorityUpdateOperatorMin::Ptr op) {
        if (direction_ == "push") {
            op->is_atomic = true;
        } else {
            op->is_atomic = false;
        }
        enclosing_func_decl_->field_vector_properties_map_[mir_context_->getPriorityVectorName()] = buildLocalReadWriteFieldProperty();
    }


    void VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::AssignStmt::Ptr assign_stmt) {
        in_write_phase = true;
        assign_stmt->lhs->accept(this);
        in_write_phase = false;
        in_read_phase = true;
        assign_stmt->expr->accept(this);
        in_read_phase = false;
    }

    void VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::ReduceStmt::Ptr reduce_stmt) {
        in_read_write_phase = true;
        reduce_stmt->lhs->accept(this);
        in_read_write_phase = false;
        in_read_phase = true;
        reduce_stmt->expr->accept(this);
        in_read_phase = false;
    }


    void VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::TensorReadExpr::Ptr tensor_read) {
        if (mir::isa<mir::VarExpr>(tensor_read->target)){
            std::string target = tensor_read->getTargetNameStr();
            std::string index = tensor_read->getIndexNameStr();
            auto field_vector_prop = determineFieldVectorProperty(target, in_write_phase,
                                                                  in_read_phase, index, direction_);
            enclosing_func_decl_->field_vector_properties_map_[target] = field_vector_prop;
            tensor_read->field_vector_prop_ = field_vector_prop;
        }
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::determineFieldVectorProperty(
            std::string field_vector_name,
            bool in_write_phase,
            bool in_read_phase,
            std::string index,
            std::string direction) {

        std::string src_var_name = enclosing_func_decl_->args[0].getName();
        std::string dst_var_name = enclosing_func_decl_->args[1].getName();
        FieldVectorProperty output;

        // if it is not a read property, just return the current property (only need to detect write or read_and_write)
        if (enclosing_func_decl_->field_vector_properties_map_.find(field_vector_name)
            != enclosing_func_decl_->field_vector_properties_map_.end()) {
            FieldVectorProperty::ReadWriteType existing_readwrite_access
                    = enclosing_func_decl_->field_vector_properties_map_[field_vector_name].read_write_type;
            if (existing_readwrite_access == FieldVectorProperty::ReadWriteType::WRITE_ONLY ||
                existing_readwrite_access == FieldVectorProperty::ReadWriteType::READ_AND_WRITE) {
                return enclosing_func_decl_->field_vector_properties_map_[field_vector_name];
            }

        }


        if (index == src_var_name) {
            // operating on src
            if (direction == "pull") {
                // direction is pull
                if (in_write_phase) {
                    //write operation
                    output = buildSharedWriteFieldProperty();
                } else if (in_read_phase) {
                    // read operation
                    output = buildSharedReadFieldProperty();
                } else {
                    output = buildSharedReadWriteFieldProperty();
                }
            } else {
                //direction is push
                if (in_write_phase) {
                    output = buildLocalWriteFieldProperty();
                } else if (in_read_phase) {
                    output = buildLocalReadFieldProperty();
                } else {
                    output = buildLocalReadWriteFieldProperty();
                }
            }
        } else {
            // operating on dst
            if (direction == "pull") {
                if (in_write_phase) {
                    // write
                    output = buildLocalWriteFieldProperty();
                } else if (in_read_phase) {
                    // read
                    output = buildLocalReadFieldProperty();
                } else {
                    output = buildLocalReadWriteFieldProperty();

                }
            } else {
                if (in_write_phase) {
                    // write
                    output = buildSharedWriteFieldProperty();
                } else if (in_read_phase) {
                    // read
                    output = buildSharedReadFieldProperty();
                } else {
                    output = buildSharedReadWriteFieldProperty();
                }
            }
        }

        return output;
    }

    // factory method for building certain type of field vector property
    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildLocalWriteFieldProperty() {
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
        property.read_write_type = FieldVectorProperty::ReadWriteType::WRITE_ONLY;
        return property;
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildSharedWriteFieldProperty() {
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::SHARED;
        property.read_write_type = FieldVectorProperty::ReadWriteType::WRITE_ONLY;
        return property;
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildLocalReadFieldProperty() {
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
        property.read_write_type = FieldVectorProperty::ReadWriteType::READ_ONLY;
        return property;
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildSharedReadFieldProperty() {
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::SHARED;
        property.read_write_type = FieldVectorProperty::ReadWriteType::READ_ONLY;
        return property;
    }


    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildLocalReadWriteFieldProperty() {
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
        property.read_write_type = FieldVectorProperty::ReadWriteType::READ_AND_WRITE;
        return property;
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildSharedReadWriteFieldProperty() {
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::SHARED;
        property.read_write_type = FieldVectorProperty::ReadWriteType::READ_AND_WRITE;
        return property;
    }




}

