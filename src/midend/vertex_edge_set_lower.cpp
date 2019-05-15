//
// Created by Yunming Zhang on 7/24/17.
//

#include <graphit/midend/vertex_edge_set_lower.h>

namespace graphit {

    void VertexEdgeSetLower::lower() {

        //a stmt block that contains the initilization of global variables
        //it would be inserted at the beginning of the main function later
        mir::StmtBlock::Ptr set_initialization_block = std::make_shared<mir::StmtBlock>();

        //lowers the constant vertex and edge set declarations
        for (auto edgeset_decl : mir_context_->const_edge_sets_) {
            //replace vardecl with an assignment
            auto assign_stmt = std::make_shared<mir::AssignStmt>();
            auto var_type = edgeset_decl->type;
            auto var_name = edgeset_decl->name;
            mir::Var mir_var = mir::Var(var_name, var_type);
            mir::VarExpr::Ptr mir_var_expr = std::make_shared<mir::VarExpr>();
            mir_var_expr->var = mir_var;

            assign_stmt->lhs = mir_var_expr;

            // generate the load expression only if the initial value for edgeset expression is specified
            if (edgeset_decl->initVal != nullptr){
                auto edgeset_load_expr = mir::to<mir::EdgeSetLoadExpr>(edgeset_decl->initVal);

                auto edge_set_type = mir::to<mir::EdgeSetType>(edgeset_decl->type);
                if (edge_set_type->weight_type != nullptr) {
                    edgeset_load_expr->is_weighted_ = true;
                }
                assign_stmt->expr = edgeset_load_expr;
                mir_context_->edgeset_alloc_stmts.push_back(assign_stmt);
            }
        }

        //Actually, we might not need to support all the vertex sets, since the global vertex sets are not really
        //allocated. They merely serve as vertex numbers
//        for (auto vertexset_decl : mir_context_->const_vertex_sets_){
//            auto assign_stmt = std::make_shared<mir::AssignStmt>();
//            auto var_type = vertexset_decl->type;
//            auto var_name = vertexset_decl->name;
//            mir::Var mir_var = mir::Var(var_name, var_type);
//            mir::VarExpr::Ptr mir_var_expr = std::make_shared<mir::VarExpr>();
//            mir_var_expr->var = mir_var;
//
//            assign_stmt->lhs = mir_var_expr;
//            assign_stmt->expr = vertexset_decl->initVal;
//            set_initialization_block->insertStmtEnd(assign_stmt);
//        }

        //insert the entire block into the beginning of main function
//        mir::FuncDecl::Ptr main_func_decl = mir_context_->getMainFuncDecl();
//        main_func_decl->body->insertStmtBlockFront(set_initialization_block);
    }
}
