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



            void addExportedFunction(mir::FuncDecl::Ptr f) {
                exported_functions_list_.push_back(f);
            }


            void addFunctionFront(mir::FuncDecl::Ptr f) {
                functions_map_[f->name] = f;
                functions_list_.insert(functions_list_.begin(), f);
            }

        // Maybe this would result in errors in the OG code
//        void addExternFunction(mir::FuncDecl::Ptr f) {
//            extern_functions_map_[f->name] = f;
//            extern_functions_list_.push_back(f);
//        }

            void addExternFunction(mir::FuncDecl::Ptr f) {
                extern_functions_map_[f->name] = f;
                extern_functions_list_.push_back(f);
                //Also add the extern function to regular function map
                //Hopefully this is not going to be an issue
                addFunction(f);
            }

            bool isExternFunction(std::string function_name){
                return extern_functions_map_.find(function_name) != extern_functions_map_.end();
            }


        bool isFunction(const std::string &name) const {
            return functions_map_.find(name) != functions_map_.end() ||
                   extern_functions_map_.find(name) != extern_functions_map_.find(name);
        }

        mir::FuncDecl::Ptr getFunction(const std::string &name) {
            assert(isFunction(name));
            return functions_map_[name];
        }


        mir::FuncDecl::Ptr getExternFunction(const std::string &name) {
            assert(isExternFunction(name));
            return extern_functions_map_[name];
        }

        void addTypeRequiringTypeDef(mir::Type::Ptr type){
            types_requiring_typedef.push_back(type);
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

        const std::vector<mir::FuncDecl::Ptr> &getExternFunctionList() {
            return extern_functions_list_;
        }

        bool hasSymbol(const std::string &name) {
            return symbol_table_.contains(name);
        }

        const mir::Var &getSymbol(const std::string &name) {
            return symbol_table_.get(name);
        }


        void addConstant(mir::VarDecl::Ptr var_decl) {
            constants_.push_back(var_decl);
        }

        std::vector<mir::VarDecl::Ptr> getConstants() {
            return constants_;
        }

        void addLoweredConstant(mir::VarDecl::Ptr var_decl) {
            lowered_constants_.push_back(var_decl);
        }

        std::vector<mir::VarDecl::Ptr> getLoweredConstants() {
            return lowered_constants_;
        }


        void addElementType(mir::ElementType::Ptr element_type) {
            //set up entries in the relevant maps
            const auto zero_int = std::make_shared<mir::IntLiteral>();
            zero_int->val = 0;

            num_elements_map_[element_type->ident] = zero_int;
            properties_map_[element_type->ident] = new std::vector<mir::VarDecl::Ptr>();
        }

        void addEdgeSet(mir::VarDecl::Ptr edgeset) {
            const_edge_sets_.push_back(edgeset);
        }

        std::vector<mir::VarDecl::Ptr> getEdgeSets() {
            return const_edge_sets_;
        }

        mir::VarDecl::Ptr getConstEdgeSetByName(std::string var_name) {

            for (auto edgeset : const_edge_sets_) {
                if (edgeset->name == var_name)
                    return edgeset;
            }
            return nullptr;
        }

        bool isConstVertexSet(std::string var_name) {
            for (auto vertexset : const_vertex_sets_) {
                if (vertexset->name == var_name) return true;
            }
            return false;
        }

        void addConstVertexSet(mir::VarDecl::Ptr vertexset) {
            const_vertex_sets_.push_back(vertexset);
            // indicate theat this element type is a vertex element type
            mir::VertexSetType::Ptr vertex_set_type = mir::to<mir::VertexSetType>(vertexset->type);
            addVertexElementType(vertex_set_type->element->ident);
        }

        std::vector<mir::VarDecl::Ptr> getConstVertexSets() {
            return const_vertex_sets_;
        }

        bool isEdgeSet(std::string var_name) {
            for (auto edgeset : const_edge_sets_) {
                if (edgeset->name == var_name) return true;
            }
            return false;
        }

        void updateVectorItemType(std::string vector_name, mir::Type::Ptr item_type) {
            vector_item_type_map_[vector_name] = item_type;
        }

        mir::Type::Ptr getVectorItemType(std::string vector_name) {
            return vector_item_type_map_[vector_name];
        }

        void addEdgesetType(std::string edgeset_name, mir::EdgeSetType::Ptr edgeset_type) {
            edgeset_element_type_map_[edgeset_name] = edgeset_type;
        }

        mir::EdgeSetType::Ptr getEdgesetType(std::string edgeset_name) {
            return edgeset_element_type_map_[edgeset_name];
        }

        bool updateElementCount(mir::ElementType::Ptr element_type, mir::Expr::Ptr count_expr) {
            if (num_elements_map_.find(element_type->ident) == num_elements_map_.end()) {
                //map does not contain the element type
                return false;
            } else {
                num_elements_map_[element_type->ident] = count_expr;
                return true;
            }
        }

        bool updateElementProperties(mir::ElementType::Ptr element_type, mir::VarDecl::Ptr var_decl) {
            if (properties_map_.find(element_type->ident) == properties_map_.end()) {
                //map does not contain the element type
                return false;
            } else {
                properties_map_[element_type->ident]->push_back(var_decl);
                setElementTypeWithVectorOrSetName(var_decl->name, element_type);
                return true;
            }
        }


        void setElementTypeWithVectorOrSetName(std::string vectorOrSetName, mir::ElementType::Ptr element_type){
		// This map is not complete right now. Elements are not being added always like arguments to functione etc. Also this is not scoped
		// So we will just use the symbol table to get the type and element type from there
		// We will still do the insertion here, but this value will never be used
		    vector_set_element_type_map_[vectorOrSetName] = element_type;
        };


        mir::ElementType::Ptr getElementTypeFromVectorOrSetName(std::string vector_name){
            // This map is not complete right now. Elements are not being added always like arguments to functione etc. Also this is not scoped
            // So we will just use the symbol table to get the type and element type from there
                    //return vector_set_element_type_map_[vector_name];
            auto var = getSymbol(vector_name);
            auto type = var.getType();
            if (mir::isa<mir::VectorType>(type))
                return mir::to<mir::VectorType>(type)->element_type;
            else if (mir::isa<mir::EdgeSetType>(type))
                return mir::to<mir::EdgeSetType>(type)->element;
            else if (mir::isa<mir::VertexSetType>(type))
                return mir::to<mir::VertexSetType>(type)->element;
            else {
                assert(false && "Cannot indentify type of vector or set\n");
            }
        }

        bool updateElementInputFilename(mir::ElementType::Ptr element_type, mir::Expr::Ptr file_name) {
            input_filename_map_[element_type->ident] = file_name;
            return true;
        }

        mir::Expr::Ptr getElementInputFilename(mir::ElementType::Ptr element_type) {
            if (input_filename_map_.find(element_type->ident) == input_filename_map_.end()) {
                return nullptr;
            } else {
                return input_filename_map_[element_type->ident];
            }
        }


        mir::Expr::Ptr getElementCount(mir::ElementType::Ptr element_type) {
            if (num_elements_map_.find(element_type->ident) == num_elements_map_.end()) {
                return nullptr;
            } else {
                return num_elements_map_[element_type->ident];
            }
        }

        std::string getUniqueNameCounterString() {
            return std::to_string(unique_variable_name_counter_++);
        }

        void insertFuncDeclFront(mir::FuncDecl::Ptr func_decl) {
            functions_list_.insert(functions_list_.begin(), func_decl);
            functions_map_[func_decl->name] = func_decl;
        }

        // Keeps track of element types that are vertex element types
        void addVertexElementType(std::string element_type) {
            vertex_element_type_list_.push_back(element_type);
        }

        // Check if an element type is a vertex element type
        // Useful for generating system vector lower operations
        bool isVertexElementType(std::string element_type) {
            for (auto vertex_element_type : vertex_element_type_list_) {
                if (vertex_element_type == element_type) return true;
            }
            return false;
        }

        void insertNewConstVectorDeclEnd(mir::VarDecl::Ptr mir_var_decl) {
            mir::VectorType::Ptr type = std::dynamic_pointer_cast<mir::VectorType>(mir_var_decl->type);
            if (type->element_type != nullptr) {
                // this is a field / system vector associated with an ElementType
                updateVectorItemType(mir_var_decl->name, type->vector_element_type);
                if (!updateElementProperties(type->element_type, mir_var_decl))
                    std::cout << "error in adding constant" << std::endl;
            }
        }

        mir::VarDecl::Ptr getGlobalConstVertexSet() {
            return const_vertex_sets_[0];
        }

        mir::FuncDecl::Ptr getMainFuncDecl() {
            for (auto &func_decl : functions_list_) {
                if (func_decl->name == "main")
                    return func_decl;
            }
            std::cout << "No main function declared" << std::endl;
            return nullptr;
        }

        mir::VarDecl::Ptr getPriorityQueueDecl(){
            for (mir::VarDecl::Ptr constant : getConstants()){
                if (mir::isa<mir::PriorityQueueType>(constant->type)){
                    return constant;
                }
            }
            return nullptr;
        }

        std::string getPriorityVectorName(){
            return priority_queue_alloc_list_[0]->vector_function;
        }

	std::vector<mir::PriorityQueueAllocExpr::Ptr> priority_queue_alloc_list_;

        //private:

        // maps element type to an input file that reads the set from
        // for example, reading an edge set
        std::map<std::string, mir::Expr::Ptr> input_filename_map_;
        // maps element type to the number of elements (initially)
        std::map<std::string, mir::Expr::Ptr> num_elements_map_;
        // maps element type to the fields associated with the type
        std::map<std::string, std::vector<mir::VarDecl::Ptr> *> properties_map_;

        // const vertex_sets and edge_sets
        // These are global sets that are loaded from outside sources and cannot be modified
        std::vector<mir::VarDecl::Ptr> const_vertex_sets_;
        std::vector<mir::VarDecl::Ptr> const_edge_sets_;

        //maps a vector to the Element it is associated with;
        std::map<std::string, mir::ElementType::Ptr> vector_set_element_type_map_;
        // maps a vector reference to item type
        std::map<std::string, mir::Type::Ptr> vector_item_type_map_;

        //maps a edgeset name to its edgeset type
        std::map<std::string, mir::EdgeSetType::Ptr> edgeset_element_type_map_;

        //private:

        std::vector<std::string> vertex_element_type_list_;
        std::vector<std::string> edge_element_type_list_;


        // constants declared in the FIR, before lowering
        std::vector<mir::VarDecl::Ptr> constants_;
        // struct declarations
        std::map<std::string, mir::StructTypeDecl::Ptr> struct_type_decls;
        // constants after the physical data layout lower pass
        std::vector<mir::VarDecl::Ptr> lowered_constants_;

        std::map<std::string, mir::FuncDecl::Ptr> functions_map_;
        std::map<std::string, mir::FuncDecl::Ptr> extern_functions_map_;

        //need to store the ordering of function declarations
        std::vector<mir::FuncDecl::Ptr> functions_list_;
        std::vector<mir::FuncDecl::Ptr> extern_functions_list_;

        // symbol table
        util::ScopedMap<std::string, mir::Var> symbol_table_;

        int unique_variable_name_counter_ = 0;

        // these vectors are used in code gen for main functions (there's a condition in func_decl code gen)
        std::vector<mir::Stmt::Ptr> edgeset_alloc_stmts;
        std::vector<mir::Stmt::Ptr> field_vector_alloc_stmts;
        std::vector<mir::Stmt::Ptr> field_vector_init_stmts;


        // used by cache/numa optimization
        std::map<std::string, std::map<std::string, int>> edgeset_to_label_to_num_segment;

        std::vector<mir::FuncDecl::Ptr> exported_functions_list_;



        // Ordered GraphIt additions

        // By default the priority is ReduceBeforePriorityUpdate
        mir::PriorityUpdateType priority_update_type = mir::PriorityUpdateType::ReduceBeforePriorityUpdate;


        int delta_ = 1;
        int bucket_merge_threshold_ = 0;
        int num_open_buckets = 128;
        bool nodes_init_in_buckets = false; // wether the nodes are initialized with values and inserted in buckets
        mir::Expr::Ptr optional_starting_source_node = nullptr;
        std::string eager_priority_update_edge_function_name = "";


        // used by numa optimization
        std::map<std::string, std::map<std::string, mir::MergeReduceField::Ptr>> edgeset_to_label_to_merge_reduce;

        std::set<std::string> defined_types;

        std::vector<mir::Type::Ptr> types_requiring_typedef;


    };

}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
