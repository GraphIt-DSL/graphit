//
// Created by Yunming Zhang on 5/10/17.
//
#include <graphit/midend/physical_data_layout_lower.h>

namespace graphit {


    void PhysicalDataLayoutLower::lower() {

        //generate variable declarations
        genVariableDecls();

        //reset the tensor reads
        auto lower_tensor_read = LowerTensorRead(schedule_);
        auto lower_vertexset_layout = LowerVertexsetDecl(schedule_);
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

        for (auto stmt : mir_context_->field_vector_init_stmts) {
            lower_tensor_read.rewrite(stmt);
        }

        for (auto stmt : mir_context_->edgeset_alloc_stmts) {
            lower_tensor_read.rewrite(stmt);
        }


        for (auto function : functions) {
            lower_tensor_read.rewrite(function);
            function->accept(&lower_vertexset_layout);
        }


    }

    void PhysicalDataLayoutLower::genVariableDecls() {


        // for each field / system vector of the element
        // lower into an IR node with physical data layout information
        for (auto const &element_type_entry : mir_context_->properties_map_) {
            // for each element type
            for (auto const &var_decl : *element_type_entry.second) {

                auto var_name = var_decl->name;


                if (schedule_ != nullptr && schedule_->physical_data_layouts != nullptr) {
                    auto physical_data_layout = schedule_->physical_data_layouts->find(var_name);

                    if (physical_data_layout != schedule_->physical_data_layouts->end()) {
                        //if physical layout schedule is being specified
                        if (physical_data_layout->second.data_layout_type == FieldVectorDataLayoutType::STRUCT) {
                            genStructDecl(var_decl, physical_data_layout->second);
                            continue;
                        }
                    }
                }

                //By default, we generate dense array implementations
                genArrayDecl(var_decl);

            }
        }

        // process non property constants
        for (auto constant : mir_context_->getConstants()) {
            mir_context_->addLoweredConstant(constant);
        }
    }

    void PhysicalDataLayoutLower::genStructDecl(const mir::VarDecl::Ptr var_decl,
                                                const FieldVectorPhysicalDataLayout data_layout) {
        // check if the struct type has already been added


        if (mir_context_->struct_type_decls.find(data_layout.fused_struct_name)
            == mir_context_->struct_type_decls.end()) {
            // add a struct type declaration and vector declaration if not added before
            auto struct_type_decl = std::make_shared<mir::StructTypeDecl>();
            mir_context_->struct_type_decls[data_layout.fused_struct_name] = struct_type_decl;
            auto array_of_struct_decl = std::make_shared<mir::VarDecl>();
            array_of_struct_decl->name = "array_of_" + data_layout.fused_struct_name;

            //initialize the struct type for the array of struct declaration
            //including its vector_element_type, associated element_type (Vertex, Edge),
            auto array_of_struct_type = std::make_shared<mir::VectorType>();
            array_of_struct_type->vector_element_type = struct_type_decl;
            auto vector_type = mir::to<mir::VectorType>(var_decl->type);
            array_of_struct_type->element_type = vector_type->element_type;

            //set the type of each struct in the array of struct
            array_of_struct_decl->type = array_of_struct_type;
            mir_context_->addLoweredConstant(array_of_struct_decl);
        }

        // initialize the value of the field
        auto struct_type_decl = mir_context_->struct_type_decls[data_layout.fused_struct_name];
        struct_type_decl->name = data_layout.fused_struct_name;

        //currently we don't support struct as field of a struct
        auto field_var_decl = std::make_shared<mir::VarDecl>();
        field_var_decl->name = var_decl->name;
        field_var_decl->initVal = var_decl->initVal;
        field_var_decl->type = mir::to<mir::VectorType>(var_decl->type)->vector_element_type;

        struct_type_decl->fields.push_back(field_var_decl);

    }

    void PhysicalDataLayoutLower::genArrayDecl(const mir::VarDecl::Ptr var_decl) {
        mir_context_->addLoweredConstant(var_decl);
    }

    /**
     * Reads the schedule and generate concrete tensor read type
     * It generates TensorArrayReadExpr when no schedule is specified or ARRAY is explicitly specified
     * It generates TensorStructReadExpr or TensorDictReadExpr when the appropriate scheule is supplied
     * @param tensor_read
     */
    void PhysicalDataLayoutLower::LowerTensorRead::visit(mir::TensorReadExpr::Ptr tensor_read) {

        if (mir::isa<mir::VarExpr>(tensor_read->target)) {
            auto target_expr = mir::to<mir::VarExpr>(tensor_read->target);
            auto target_name = target_expr->var.getName();

            if (schedule_ != nullptr && schedule_->physical_data_layouts != nullptr) {
                auto physical_data_layout = schedule_->physical_data_layouts->find(target_name);

                if (physical_data_layout != schedule_->physical_data_layouts->end()) {
                    if (physical_data_layout->second.data_layout_type == FieldVectorDataLayoutType::STRUCT) {
                        //Generate TensorStructReadExpr
                        auto tensor_struct_read = std::make_shared<mir::TensorStructReadExpr>();
                        tensor_struct_read->index = rewrite<mir::Expr>(tensor_read->index);
                        tensor_struct_read->field_target = rewrite<mir::Expr>(tensor_read->target);
                        tensor_struct_read->array_of_struct_target =
                                "array_of_" + physical_data_layout->second.fused_struct_name;
                        tensor_struct_read->field_vector_prop_ = tensor_read->field_vector_prop_;
                        node = tensor_struct_read;
                        return;
                    }
                }
            }
        }
        //Generate TensorArrayReadExpr for no schedule specified or ARRAY data layout type
        // or when the target is another tensor expression
        auto tensor_array_read = std::make_shared<mir::TensorArrayReadExpr>();
        tensor_array_read->index = rewrite<mir::Expr>(tensor_read->index);
        tensor_array_read->target = rewrite<mir::Expr>(tensor_read->target);
        tensor_array_read->field_vector_prop_ = tensor_read->field_vector_prop_;
        node = tensor_array_read;


    }

    void PhysicalDataLayoutLower::LowerVertexsetDecl::visit(mir::VarDecl::Ptr var_decl) {
        if (mir::isa<mir::VertexSetType>(var_decl->type)) {
            //attach scheduling labels to vertexset declarations
            if (mir::isa<mir::VertexSetAllocExpr>(var_decl->initVal)) {
                //try to supply new arguments to vertexset constructor
                auto vertexset_alloc_expr = mir::to<mir::VertexSetAllocExpr>(var_decl->initVal);
                if (schedule_ != nullptr
                    &&
                    schedule_->vertexset_data_layout.find(var_decl->name) != schedule_->vertexset_data_layout.end()) {
                    //TODO: involve label scope, right now, this vertexset assumes that it would be in the root scope
                    switch (schedule_->vertexset_data_layout[var_decl->name].data_layout_type) {
                        case VertexsetPhysicalLayout::DataLayout::SPARSE :
                            vertexset_alloc_expr->layout = mir::VertexSetAllocExpr::Layout::SPARSE;
                            break;
                        case VertexsetPhysicalLayout::DataLayout::DENSE:
                            vertexset_alloc_expr->layout = mir::VertexSetAllocExpr::Layout::DENSE;
                            break;

                    }
                }
            }
        }
    }
}
