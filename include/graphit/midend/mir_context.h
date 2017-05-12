//
// Created by Yunming Zhang on 2/12/17.
//

#ifndef GRAPHIT_PROGRAM_CONTEXT_H
#define GRAPHIT_PROGRAM_CONTEXT_H

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <utility>

#include <graphit/utils/scopedmap.h>
#include <graphit/midend/mir.h>

namespace graphit {


        // Data structure that holds the internal representation of the program
        class MIRContext {

        public:
            MIRContext() {
            }


            ~MIRContext() {
            }

            void scope() {
                symbol_table_.scope();
            }

            void unscope() {
                symbol_table_.unscope();
            }

            void addFunction(mir::FuncDecl::Ptr f) {
                functions_map_[f->name] = f;
                functions_list_.push_back(f);
            }



            bool containsFunction(const std::string &name) const {
                return functions_map_.find(name) != functions_map_.end();
            }

            mir::FuncDecl::Ptr getFunction(const std::string &name) {
                assert(containsFunction(name));
                return functions_map_[name];
            }

                void addSymbol(mir::Var var) {
                symbol_table_.insert(var.getName(), var);
            }

            const std::map<std::string, mir::FuncDecl::Ptr> &getFunctionMap() {
                return functions_map_;
            }

            const std::vector<mir::FuncDecl::Ptr> &getFunctionList() {
                return functions_list_;
            }

            bool hasSymbol(const std::string &name) {
                return symbol_table_.contains(name);
            }

            const mir::Var &getSymbol(const std::string &name) {
                return symbol_table_.get(name);
            }


            void addConstant(mir::VarDecl::Ptr var_decl){
                constants_.push_back(var_decl);
            }

            std::vector<mir::VarDecl::Ptr> getConstants(){
                return constants_;
            }

            void addLoweredConstant(mir::VarDecl::Ptr var_decl){
                lowered_constants_.push_back(var_decl);
            }

            std::vector<mir::VarDecl::Ptr> getLoweredConstants(){
                return lowered_constants_;
            }


            void addElementType(mir::ElementType::Ptr element_type){
                //set up entries in the relevant maps
                const auto zero_int = std::make_shared<mir::IntLiteral>();
                zero_int->val = 0;

                num_elements_map_[element_type->ident] = zero_int;
                properties_map_[element_type->ident] = new std::vector<mir::VarDecl::Ptr>();
            }

            void addEdgeSet(mir::VarDecl::Ptr edgeset){
                edge_sets_.push_back(edgeset);
            }

            std::vector<mir::VarDecl::Ptr> getEdgeSets(){
                return edge_sets_;
            }

            bool isVertexSet(std::string var_name){
                for (auto vertexset : vertex_sets_) {
                    if (vertexset->name == var_name) return  true;
                }
                return false;
            }

            void addVertexSet(mir::VarDecl::Ptr vertexset){
                vertex_sets_.push_back(vertexset);
            }

            std::vector<mir::VarDecl::Ptr> getVertexSets(){
                return vertex_sets_;
            }

            bool isEdgeSet(std::string var_name){
                for (auto edgeset : edge_sets_) {
                    if (edgeset->name == var_name) return  true;
                }
                return false;
            }

            void updateVectorItemType(std::string vector_name, mir::Type::Ptr item_type){
                vector_item_type_map_[vector_name] = item_type;
            }

            mir::Type::Ptr getVectorItemType(std::string vector_name){
                return vector_item_type_map_[vector_name];
            }

            bool updateElementCount(mir::ElementType::Ptr element_type, mir::Expr::Ptr count_expr){
                if (num_elements_map_.find(element_type->ident) == num_elements_map_.end()){
                    //map does not contain the element type
                    return false;
                } else {
                    num_elements_map_[element_type->ident] = count_expr;
                    return true;
                }
            }

            bool updateElementProperties(mir::ElementType::Ptr element_type, mir::VarDecl::Ptr var_decl){
                if (properties_map_.find(element_type->ident) == properties_map_.end()){
                    //map does not contain the element type
                    return false;
                } else {
                    properties_map_[element_type->ident]->push_back(var_decl);
                    setElementTypeWithVectorOrSetName(var_decl->name, element_type);
                    return true;
                }
            }

            void setElementTypeWithVectorOrSetName(std::string vectorOrSetName, mir::ElementType::Ptr element_type){
                vector_set_element_type_map_[vectorOrSetName] = element_type;

            };

            mir::ElementType::Ptr getElementTypeFromVectorOrSetName(std::string vector_name){
                return vector_set_element_type_map_[vector_name];
            }

            bool updateElementInputFilename(mir::ElementType::Ptr  element_type, mir::Expr::Ptr file_name){
                input_filename_map_[element_type->ident] = file_name;
                return true;
            }

            mir::Expr::Ptr getElementInputFilename(mir::ElementType::Ptr element_type){
                if (input_filename_map_.find(element_type->ident) == input_filename_map_.end()) {
                    return nullptr;
                } else {
                    return input_filename_map_[element_type->ident];
                }
            }


            mir::Expr::Ptr getElementCount(mir::ElementType::Ptr element_type){
                if (num_elements_map_.find(element_type->ident) == num_elements_map_.end()) {
                    return nullptr;
                } else {
                    return num_elements_map_[element_type->ident];
                }
            }

        //private:

            // maps element type to an input file that reads the set from
            // for example, reading an edge set
            std::map<std::string, mir::Expr::Ptr> input_filename_map_;
            // maps element type to the number of elements (initially)
            std::map<std::string, mir::Expr::Ptr> num_elements_map_;
            // maps element type to the fields associated with the type
            std::map<std::string, std::vector<mir::VarDecl::Ptr>*> properties_map_;

            //vertex_sets and edge_sets
            std::vector<mir::VarDecl::Ptr> vertex_sets_;
            std::vector<mir::VarDecl::Ptr> edge_sets_;

            //maps a vector to the Element it is associated with;
            std::map<std::string, mir::ElementType::Ptr> vector_set_element_type_map_;
            // maps a vector reference to item type
            std::map<std::string, mir::Type::Ptr> vector_item_type_map_;

            // constants declared in the FIR, before lowering
            std::vector<mir::VarDecl::Ptr> constants_;
            // struct declarations
            std::map<std::string, mir::StructTypeDecl::Ptr> struct_type_decls;
            // constants after the physical data layout lower pass
            std::vector<mir::VarDecl::Ptr> lowered_constants_;

            std::map<std::string, mir::FuncDecl::Ptr>  functions_map_;
            //need to store the ordering of function declarations
            std::vector<mir::FuncDecl::Ptr> functions_list_;

            // symbol table
            util::ScopedMap<std::string, mir::Var> symbol_table_;

        };

}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
