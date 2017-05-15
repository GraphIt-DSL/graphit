//
// Created by Yunming Zhang on 5/10/17.
//
#include <graphit/midend/physical_data_layout_lower.h>
namespace  graphit {


    void PhysicalDataLayoutLower::lower() {

        //generate variable declarations
        genVariableDecls();

        //reset the tensor reads
        auto lower_tensor_read = LowerTensorRead(schedule_);

        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

        for (auto function : functions){
            lower_tensor_read.rewrite(function);
        }


    }

    void PhysicalDataLayoutLower::genVariableDecls() {


        // for each field / system vector of the element
        // lower into an IR node with physical data layout information
        for (auto const & element_type_entry : mir_context_->properties_map_){
            // for each element type
            for (auto const & var_decl : *element_type_entry.second){

                auto var_name = var_decl->name;


                if(schedule_ !=nullptr && schedule_->physical_data_layouts != nullptr){
                    auto physical_data_layout = schedule_->physical_data_layouts->find(var_name);

                    if (physical_data_layout != schedule_->physical_data_layouts->end()){
                        //if physical layout schedule is being specified
                        if (physical_data_layout->second.data_layout_type == DataLayoutType::STRUCT){
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
        for (auto constant : mir_context_->getConstants()){
            mir_context_->addLoweredConstant(constant);
        }
    }

    void PhysicalDataLayoutLower::genStructDecl(const mir::VarDecl::Ptr var_decl,
                                                const PhysicalDataLayout data_layout) {
        // check if the struct type has already been added


        if (mir_context_->struct_type_decls.find(data_layout.fused_struct_name)
                == mir_context_->struct_type_decls.end()){
            // add a struct type declaration and vector declaration if not added before
            auto struct_type_decl = std::make_shared<mir::StructTypeDecl>();
            mir_context_->struct_type_decls[data_layout.fused_struct_name] = struct_type_decl;
            auto array_of_struct_decl = std::make_shared<mir::VarDecl>();
            array_of_struct_decl->name = "array_of_" + data_layout.fused_struct_name;

            //initialize the struct type for the array of struct declaration
            //including its vector_element_type, associated element_type (Vertex, Edge),
            auto array_of_struct_type = std::make_shared<mir::VectorType>();
            array_of_struct_type->vector_element_type = struct_type_decl;
            auto vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
            assert(vector_type != nullptr);
            array_of_struct_type->element_type = vector_type->element_type;

            //set the type of each struct in the array of struct
            array_of_struct_decl->type = array_of_struct_type;
            mir_context_->addLoweredConstant(array_of_struct_decl);
        }

        // initialize the value of the field
        auto struct_type_decl = mir_context_->struct_type_decls[data_layout.fused_struct_name];
        struct_type_decl->name = data_layout.fused_struct_name;

        //currently we don't support struct as field of astruct
        auto field_var_decl = std::make_shared<mir::VarDecl>();
        field_var_decl->name = var_decl->name;
        field_var_decl->initVal = var_decl->initVal;
        field_var_decl->type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type)->vector_element_type;

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
        //make no changes to the default array implementation
        auto tensor_array_read = std::make_shared<mir::TensorArrayReadExpr>();
        tensor_array_read->index = rewrite<mir::Expr>(tensor_read->index);
        tensor_array_read->target = rewrite<mir::Expr>(tensor_read->target);
        node = tensor_array_read;
    }

}
