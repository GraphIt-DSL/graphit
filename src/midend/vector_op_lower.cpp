//
// Created by Yunming Zhang on 6/22/17.
//

#include <graphit/midend/vector_op_lower.h>

namespace  graphit {

    void VectorOpLower::lowerConstVectorVarDecl() {
        // for each field / system vector of the element
        // lower into an IR node with physical data layout information
        for (auto const &element_type_entry :mir_context_->properties_map_) {
            // for each element type\
            // making a copy of the properties for element_type_entry
            // since we might be inserting into it along the way
            auto element_properties_copy = *element_type_entry.second;
            for (auto const &var_decl : element_properties_copy) {
                if (mir::isa<mir::Call>(var_decl->initVal)){
                    auto orig_init_val = var_decl->initVal;
                    mir::VectorType::Ptr vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);

                    if (mir::isa<mir::ScalarType>(vector_type->vector_element_type)){
                        mir::ScalarType::Ptr element_type =  mir::to<mir::ScalarType>(
                                vector_type->vector_element_type);
                        //reset the initval to something default 0 for integer and float
                        if (element_type->type == mir::ScalarType::Type::INT){
                            //initial value should be a int
                            auto zero = std::make_shared<mir::IntLiteral>();
                            zero->val = 0;
                            var_decl->initVal = zero;
                        }
                        else if (element_type->type == mir::ScalarType::Type::FLOAT){
                            //initial value should be a float

                        }

                        //insert another const var decl as the temporary holder for the function
                        auto tmp_var_decl = std::make_shared<mir::VarDecl>();
                        tmp_var_decl->type = var_decl->type;
                        tmp_var_decl->initVal = orig_init_val;
                        tmp_var_decl->name = "generated_tmp_vector_" + mir_context_->getUniqueNameCounterString();
                        tmp_var_decl->modifier = var_decl->modifier;
                        mir_context_->insertNewConstVectorDeclEnd(tmp_var_decl);

                        //create a new apply function decl that copies over the vector
                        if (mir_context_->isVertexElementType(vector_type->element_type->ident)){
                            //a vertexset apply function if the element is a vertexset
                            mir::FuncDecl::Ptr copy_over_apply_func = std::make_shared<mir::FuncDecl>();
                            // create a utility function for creating new vertexset apply
                            // set up a name
                            copy_over_apply_func->name = "generated_vector_op_apply_func_"
                                                         + mir_context_->getUniqueNameCounterString();
                            auto arg_var_type = vector_type->element_type;
                            mir::Var arg_var = mir::Var("v", arg_var_type);
                            std::vector<mir::Var> arg_var_list = std::vector<mir::Var>();
                            arg_var_list.push_back(arg_var);
                            copy_over_apply_func->args = arg_var_list;

                            auto mir_stmt_body = std::make_shared<mir::StmtBlock>();
                            auto assign_stmt = std::make_shared<mir::AssignStmt>();

                            auto lhs = std::make_shared<mir::TensorReadExpr>(
                                    var_decl->name, "v",
                                    var_decl->type,
                                    vector_type->element_type
                                    );

                            auto rhs = std::make_shared<mir::TensorReadExpr>(
                                    tmp_var_decl->name, "v",
                                    tmp_var_decl->type,
                                    vector_type->element_type
                            );

                            assign_stmt->lhs = lhs;
                            assign_stmt->expr = rhs;
                            mir_stmt_body->insertStmtEnd(assign_stmt);
                            copy_over_apply_func->body = mir_stmt_body;
                            //insert the utility function back into function list
                            mir_context_->insertFuncDeclFront(copy_over_apply_func);


                            // Lastly, insert a vertexset apply expression at the beginning of main
                            mir::VarDecl::Ptr global_vertex_set_var_decl = mir_context_->getGlobalConstVertexSet();
                            mir::VertexSetApplyExpr::Ptr vertex_set_apply_expr =
                                    std::make_shared<mir::VertexSetApplyExpr>(global_vertex_set_var_decl->name,
                                                                              global_vertex_set_var_decl->type,
                                                                              copy_over_apply_func->name);
                            mir::ExprStmt::Ptr apply_stmt = std::make_shared<mir::ExprStmt>();
                            apply_stmt->expr = vertex_set_apply_expr;
                            mir::FuncDecl::Ptr main_func_decl = mir_context_->getMainFuncDecl();
                            main_func_decl->body->insertStmtFront(apply_stmt);
                        }

                    }

                }
            }
        }
    }

    void VectorOpLower::lower() {
        lowerConstVectorVarDecl();
        //TODO: lower other vector operations
    }
}