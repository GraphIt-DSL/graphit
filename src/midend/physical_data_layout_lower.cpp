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
                            genStructDecl(var_decl);
                            continue;
                        }
                    }
                }

                genArrayDecl(var_decl);

            }
        }
    }

    void PhysicalDataLayoutLower::genStructDecl(const mir::VarDecl::Ptr &shared_ptr) {

    }

    void PhysicalDataLayoutLower::genArrayDecl(const mir::VarDecl::Ptr &shared_ptr) {

    }
}
