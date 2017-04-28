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
                //symbol_table.scope();
                statements_.push_front(std::vector<mir::Stmt::Ptr>());
                //builder.setInsertionPoint(&statements.front());
            }

            void unscope() {
                //symbol_table.unscope();
                statements_.pop_front();
                //builder.setInsertionPoint(statements.size() > 0 ? &statements.front() : nullptr);
            }

            void addFunction(mir::FuncDecl::Ptr f) {
                functions_[f->name] = f;
            }

            bool containsFunction(const std::string &name) const {
                return functions_.find(name) != functions_.end();
            }

            mir::FuncDecl::Ptr getFunction(const std::string &name) {
                assert(containsFunction(name));
                return functions_[name];
            }

                void addSymbol(mir::Var var) {
                symbol_table_.insert(var.getName(), var);
            }

            const std::map<std::string, mir::FuncDecl::Ptr> &getFunctions() {
                return functions_;
            }

            bool hasSymbol(const std::string &name) {
                return symbol_table_.contains(name);
            }

            const mir::Var &getSymbol(const std::string &name) {
                return symbol_table_.get(name);
            }


            //void setProgram(mir::Stmt::Ptr program){this->mir_program = program};
            void addConstant(mir::VarDecl::Ptr var_decl){
                constants_.push_back(var_decl);
            }

            std::vector<mir::VarDecl::Ptr> getConstants(){
                return constants_;
            }

            void addStatement(mir::Stmt::Ptr stmt) {
                statements_.front().push_back(stmt);
            }

            std::vector<mir::Stmt::Ptr> * getStatements() {
                return &statements_.front();
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

            void updateVectorItemType(std::string vector_name, mir::ScalarType::Ptr item_type){
                vector_item_type_map_[vector_name] = item_type;
            }

            mir::ScalarType::Ptr getVectorItemType(std::string vector_name){
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

            //mir::Program::Ptr mir_program;

            //maps a vector reference to its physical layout in the current scope
            util::ScopedMap<std::string, std::string> layout_map_;
            std::vector<mir::VarDecl::Ptr> edge_sets_;
            //maps a vector to the Element it is associated with;
            std::map<std::string, mir::ElementType::Ptr> vector_set_element_type_map_;
            // maps a vector reference to item type
            std::map<std::string, mir::ScalarType::Ptr> vector_item_type_map_;
            // maps element type to an input file that reads the set from
            std::map<std::string, mir::Expr::Ptr> input_filename_map_;
            // maps element type to the number of elements (initially)
            std::map<std::string, mir::Expr::Ptr> num_elements_map_;
            // maps element type to the fields associated with the type
            std::map<std::string, std::vector<mir::VarDecl::Ptr>*> properties_map_;
            std::vector<mir::VarDecl::Ptr> constants_;
            std::list<std::vector<mir::Stmt::Ptr>> statements_;
            std::map<std::string, mir::FuncDecl::Ptr>  functions_;
            util::ScopedMap<std::string, mir::Var> symbol_table_;

        };

}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
