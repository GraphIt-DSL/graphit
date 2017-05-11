//
// Created by Yunming Zhang on 5/10/17.
//
#include <graphit/midend/physical_data_layout_lower.h>
namespace  graphit {

    void PhysicalDataLayoutLower::lower() {

        //generate variable declarations
        genVariableDecls();

        //reset the tensor reads
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
            array_of_struct_decl->name = data_layout.fused_struct_name;
            array_of_struct_decl->type = struct_type_decl;
            mir_context_->addLoweredConstant(array_of_struct_decl);
        }

        // initialize the value of the field
        auto struct_type_decl = mir_context_->struct_type_decls[data_layout.fused_struct_name];
        struct_type_decl->name = data_layout.fused_struct_name;
        struct_type_decl->fields.push_back(var_decl);

    }

    void PhysicalDataLayoutLower::genArrayDecl(const mir::VarDecl::Ptr var_decl) {
        mir_context_->addLoweredConstant(var_decl);
    }
}
