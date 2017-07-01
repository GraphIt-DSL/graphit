//
// Created by Yunming Zhang on 6/29/17.
//

#include <graphit/midend/vector_field_properties_analyzer.h>

namespace graphit {

    void VectorFieldPropertiesAnalyzer::analyze() {
        auto apply_expr_visitor = ApplyExprVisitor(mir_context_, schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
        for (auto function : functions){
            function->accept(&apply_expr_visitor);
        }
    }

    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor
        ::visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr) {
        auto property_visitor = PropertyAnalyzingVisitor("pull");
        auto apply_func_decl_name = apply_expr->input_function_name;
        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_func_decl_name);
        apply_func_decl->accept(&property_visitor);

    }

    void VectorFieldPropertiesAnalyzer::ApplyExprVisitor
        ::visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr) {
        auto property_visitor = PropertyAnalyzingVisitor("push");
        auto apply_func_decl_name = apply_expr->input_function_name;
        mir::FuncDecl::Ptr apply_func_decl = mir_context_->getFunction(apply_func_decl_name);
        apply_func_decl->accept(&property_visitor);
    }

    void VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::AssignStmt::Ptr assign_stmt) {
        in_write_phase = true;
        assign_stmt->lhs->accept(this);
        in_write_phase = false;
        in_read_phase = true;
        assign_stmt->expr->accept(this);
        in_read_phase = false;
    }

    void VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::visit(mir::TensorReadExpr::Ptr tensor_read) {
            std::string target = tensor_read->getTargetNameStr();
            std::string index = tensor_read->getIndexNameStr();
            enclosing_func_decl_->field_vector_properties_map_[target] =
                    determineFieldVectorProperty(in_write_phase, in_read_phase, index, direction_);

    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::determineFieldVectorProperty(
                                                     bool in_write_phase,
                                                     bool in_read_phase,
                                                     std::string index,
                                                     std::string direction) {

        std::string src_var_name = enclosing_func_decl_->args[0].getName();
        std::string dst_var_name = enclosing_func_decl_->args[1].getName();
        FieldVectorProperty output = FieldVectorProperty();

        if (index == src_var_name) {
            // operating on src
            if (direction == "pull") {
                // direction is pull
                if (in_write_phase) {
                    //write operation
                    output = buildSharedWriteFieldProperty();
                } else {
                    // read operation
                    output = buildSharedReadFieldProperty();
                }
            } else {
                //direction is push
                if (in_write_phase) {
                    output = buildLocalWriteFieldProperty();
                } else {
                    output = buildLocalReadFieldProperty();
                }
            }
        } else {
            // operating on dst
            if (direction == "pull") {
                if (in_write_phase) {
                    // write
                    output = buildLocalWriteFieldProperty();
                } else {
                    // read
                    output = buildLocalReadFieldProperty();
                }
            } else {
                if (in_write_phase) {
                    // write
                    output = buildSharedWriteFieldProperty();
                } else {
                    // read
                    output = buildSharedReadFieldProperty();
                }
            }
        }

        return output;
    }

    // factory method for building certain type of field vector property
    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildLocalWriteFieldProperty(){
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
        property.read_write_type = FieldVectorProperty::ReadWriteType::WRITE_ONLY;
        return property;
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildSharedWriteFieldProperty(){
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::SHARED;
        property.read_write_type = FieldVectorProperty::ReadWriteType::WRITE_ONLY;
        return property;
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildLocalReadFieldProperty(){
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::LOCAL;
        property.read_write_type = FieldVectorProperty::ReadWriteType::READ_ONLY;
        return property;
    }

    FieldVectorProperty VectorFieldPropertiesAnalyzer::PropertyAnalyzingVisitor::buildSharedReadFieldProperty(){
        auto property = FieldVectorProperty();
        property.access_type_ = FieldVectorProperty::AccessType::SHARED;
        property.read_write_type = FieldVectorProperty::ReadWriteType::READ_ONLY;
        return property;
    }

}

