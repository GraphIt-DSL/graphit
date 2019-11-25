//
// Created by Yunming Zhang on 5/12/17.
//

#include <graphit/midend/mir_rewriter.h>


namespace graphit {
    namespace mir {

        void MIRRewriter::visit(Expr::Ptr expr) {
            node = rewrite<Expr>(expr);
        };


        void MIRRewriter::visit(ExprStmt::Ptr stmt) {
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }
            stmt->expr = rewrite<Expr>(stmt->expr);
            node = stmt;
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

        void MIRRewriter::visit(AssignStmt::Ptr stmt) {
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }
            stmt->lhs = rewrite<Expr>(stmt->lhs);
            stmt->expr = rewrite<Expr>(stmt->expr);
            node = stmt;
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

        void MIRRewriter::visit(CompareAndSwapStmt::Ptr stmt) {
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }
            stmt->lhs = rewrite<Expr>(stmt->lhs);
            stmt->expr = rewrite<Expr>(stmt->expr);
            stmt->compare_val_expr = rewrite<Expr>(stmt->expr);
            node = stmt;
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

        void MIRRewriter::visit(ReduceStmt::Ptr stmt) {
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }
            stmt->lhs = rewrite<Expr>(stmt->lhs);
            stmt->expr = rewrite<Expr>(stmt->expr);
            node = stmt;
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

        void MIRRewriter::visit(PrintStmt::Ptr stmt) {
            stmt->expr = rewrite<Expr>(stmt->expr);
            node = stmt;
        }

        void MIRRewriter::visit(StmtBlock::Ptr stmt_block) {
            for (int i = 0; i < (stmt_block->stmts)->size(); i++){
                auto tmp = rewrite<Stmt>(stmt_block->stmts->at(i));
                stmt_block->stmts->at(i) = tmp;
            }
            node = stmt_block;
        }


        void MIRRewriter::visit(FuncDecl::Ptr func_decl) {

            if (func_decl->body && func_decl->body->stmts) {
                func_decl->body = rewrite<StmtBlock>(func_decl->body);
            }
            node = func_decl;
        }

        void MIRRewriter::rewrite_call_args(Call::Ptr expr){
            for (auto &arg : expr->args) {
                arg = rewrite<Expr>(arg);
            }
        }

        void MIRRewriter::visit(Call::Ptr expr) {
            // need to use & to actually modify the argument (with reference), not a copy of it
            rewrite_call_args(expr);
            node = expr;
        };

        void MIRRewriter::visit(MulExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(DivExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(NegExpr::Ptr expr) {
            expr->operand = rewrite<mir::Expr>(expr->operand);
            node = expr;
        }

        void MIRRewriter::visit(EqExpr::Ptr expr) {
            visitNaryExpr(expr);
        }

        void MIRRewriter::visit(AndExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(OrExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(XorExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(NotExpr::Ptr expr) {
            expr->operand = rewrite<mir::Expr>(expr->operand);
            node = expr;
        }

        void MIRRewriter::visit(AddExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(SubExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(std::shared_ptr<VarDecl> var_decl) {

            if (var_decl->stmt_label != "") {
                label_scope_.scope(var_decl->stmt_label);
            }
            if (var_decl->initVal != nullptr) {
                var_decl->initVal = rewrite<Expr>(var_decl->initVal);
            }

            var_decl->type = rewrite<Type>(var_decl->type);

            node = var_decl;

            if (var_decl->stmt_label != "") {
                label_scope_.unscope();
            }
        }


        void MIRRewriter::visit(std::shared_ptr<VertexSetAllocExpr> expr) {
            expr->size_expr = rewrite<Expr>(expr->size_expr);
            expr->element_type = rewrite<ElementType>(expr->element_type);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<VectorAllocExpr> expr) {
            expr->size_expr = rewrite<Expr>(expr->size_expr);
	    if (expr->element_type != nullptr)
                expr->element_type = rewrite<ElementType>(expr->element_type);
            if (expr->scalar_type != nullptr)
                expr->scalar_type = rewrite<ScalarType>(expr->scalar_type);
            else if (expr->vector_type != nullptr)
                expr->vector_type = rewrite<VectorType>(expr->vector_type);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<ListAllocExpr> expr) {
            if (expr->size_expr != nullptr)
                expr->size_expr = rewrite<Expr>(expr->size_expr);
            expr->element_type = rewrite<Type>(expr->element_type);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<VertexSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<EdgeSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<PushEdgeSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<PullEdgeSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<UpdatePriorityEdgeSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<HybridDenseEdgeSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<HybridDenseForwardEdgeSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<VertexSetWhereExpr> expr) {
            //expr->input_func = rewrite<Expr>(expr->input_func);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<TensorReadExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            expr->index = rewrite<Expr>(expr->index);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<TensorStructReadExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            expr->index = rewrite<Expr>(expr->index);
            expr->field_target = rewrite<Expr>(expr->field_target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<TensorArrayReadExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            expr->index = rewrite<Expr>(expr->index);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<NameNode> stmt) {
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }
            stmt->body = rewrite<StmtBlock>(stmt->body);
            node = stmt;
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

        void MIRRewriter::visit(std::shared_ptr<ForStmt> stmt) {
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }
            stmt->domain = rewrite<ForDomain>(stmt->domain);
            stmt->body = rewrite<StmtBlock>(stmt->body);
            node = stmt;
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }

        void MIRRewriter::visit(std::shared_ptr<ForDomain> for_domain) {
            for_domain->lower = rewrite<Expr>(for_domain->lower);
            for_domain->upper = rewrite<Expr>(for_domain->upper);
            node = for_domain;
        }

        void MIRRewriter::visit(std::shared_ptr<StructTypeDecl> struct_type_decl) {
            for (auto field : struct_type_decl->fields) {
                field = rewrite<VarDecl>(field);
            }
            node = struct_type_decl;
        }

        void MIRRewriter::visit(std::shared_ptr<VertexSetType> vertexset_type) {
            vertexset_type->element = rewrite<ElementType>(vertexset_type->element);
            node = vertexset_type;
        }

        void MIRRewriter::visit(std::shared_ptr<ListType> list_type) {
            list_type->element_type = rewrite<Type>(list_type->element_type);
            node = list_type;
        }

        void MIRRewriter::visit(std::shared_ptr<EdgeSetType> edgeset_type) {
            edgeset_type->element = rewrite<ElementType>(edgeset_type->element);
            for (auto element_type : *edgeset_type->vertex_element_type_list) {
                element_type = rewrite<ElementType>(element_type);
            }
            node = edgeset_type;
        }

        void MIRRewriter::visit(std::shared_ptr<VectorType> vector_type) {
            if (vector_type->element_type != nullptr)
                vector_type->element_type = rewrite<ElementType>(vector_type->element_type);
            vector_type->vector_element_type = rewrite<Type>(vector_type->vector_element_type);
            node = vector_type;
        }

        void MIRRewriter::visitBinaryExpr(BinaryExpr::Ptr expr) {
            expr->lhs = rewrite<Expr>(expr->lhs);
            expr->rhs = rewrite<Expr>(expr->rhs);
            node = expr;
        }

        void MIRRewriter::visitNaryExpr(NaryExpr::Ptr expr) {
            // Here we are modifying the original operand, so need the & before operand
            for (auto &operand : expr->operands) {
                operand = rewrite<Expr>(operand);
            }
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<WhileStmt> stmt) {
            if (stmt->stmt_label != "") {
                label_scope_.scope(stmt->stmt_label);
            }
            stmt->body = rewrite<StmtBlock>(stmt->body);
            stmt->cond = rewrite<Expr>(stmt->cond);
            node = stmt;
            if (stmt->stmt_label != "") {
                label_scope_.unscope();
            }
        }


        void MIRRewriter::visit(IfStmt::Ptr stmt) {
            stmt->cond = rewrite<Expr>(stmt->cond);
            stmt->ifBody = rewrite<Stmt>(stmt->ifBody);
            if (stmt->elseBody) {
                stmt->elseBody = rewrite<Stmt>(stmt->elseBody);
            }
            node = stmt;
        }

        void MIRRewriter::visit(EdgeSetLoadExpr::Ptr load_expr) {
            load_expr->file_name = rewrite<Expr>(load_expr->file_name);
            node = load_expr;
        }

        void MIRRewriter::visit(IntersectNeighborExpr::Ptr inter_neigh_expr) {
            inter_neigh_expr->edges = rewrite<Expr>(inter_neigh_expr->edges);
            inter_neigh_expr->vertex_a = rewrite<Expr>(inter_neigh_expr->vertex_a);
            inter_neigh_expr->vertex_b = rewrite<Expr>(inter_neigh_expr->vertex_b);
            inter_neigh_expr->intersectionType = inter_neigh_expr->intersectionType;
            node = inter_neigh_expr;
        }


        void MIRRewriter::visit(IntersectionExpr::Ptr inter_expr) {
            inter_expr->vertex_a = rewrite<Expr>(inter_expr->vertex_a);
            inter_expr->vertex_b = rewrite<Expr>(inter_expr->vertex_b);
            inter_expr->numA = rewrite<Expr>(inter_expr->numA);
            inter_expr->numB = rewrite<Expr>(inter_expr->numB);
            if (inter_expr->reference != nullptr) {
                inter_expr->reference = rewrite<Expr>(inter_expr->reference);
            }

            inter_expr->intersectionType = inter_expr->intersectionType;
            node = inter_expr;
        }


        // OG Additions
        void MIRRewriter::visit(UpdatePriorityExternVertexSetApplyExpr::Ptr apply_expr) {
            apply_expr->target = rewrite<Expr>(apply_expr->target);
            node = apply_expr;
        }

        void MIRRewriter::visit(PriorityQueueType::Ptr queue_type) {
            queue_type->element = rewrite<ElementType>(queue_type->element);
            queue_type->priority_type = rewrite<ScalarType>(queue_type->priority_type);
            node = queue_type;
        }

        void MIRRewriter::visit(PriorityQueueAllocExpr::Ptr expr) {
            expr->element_type = rewrite<ElementType>(expr->element_type);
            expr->starting_node = rewrite<Expr>(expr->starting_node);
            expr->priority_type = rewrite<ScalarType>(expr->priority_type);
            node = expr;
        }


        void MIRRewriter::visit(UpdatePriorityUpdateBucketsCall::Ptr stmt) {
            //stmt->priority_queue = rewrite<Expr>(stmt->priority_queue);
            node = stmt;
        }

        void MIRRewriter::visit(UpdatePriorityExternCall::Ptr stmt) {
            stmt->input_set = rewrite<Expr>(stmt->input_set);
            node = stmt;
        }

        void MIRRewriter::visit(std::shared_ptr<OrderedProcessingOperator> op) {
            assert(op->while_cond_expr != nullptr);
            op->while_cond_expr = rewrite<Expr>(op->while_cond_expr);
            node = op;
        }

        void MIRRewriter::visit(std::shared_ptr<PriorityUpdateOperator> op) {
            rewrite_call_args(op);
            rewrite_priority_update_operator(op);
            node = op;
        }

        void MIRRewriter::visit(std::shared_ptr<PriorityUpdateOperatorMin> op) {
            rewrite_call_args(op);
            rewrite_priority_update_operator(op);
            op->old_val = rewrite<Expr>(op->old_val);
            op->new_val = rewrite<Expr>(op->new_val);
            node = op;
        }


        void MIRRewriter::visit(std::shared_ptr<PriorityUpdateOperatorSum> op) {
            rewrite_call_args(op);
            rewrite_priority_update_operator(op);
            op->delta = rewrite<Expr>(op->delta);
            op->minimum_val = rewrite<Expr>(op->minimum_val);
            node = op;
        }

        void MIRRewriter::rewrite_priority_update_operator(PriorityUpdateOperator::Ptr op) {
            op->destination_node_id = rewrite<Expr>(op->destination_node_id);
            op->priority_queue = rewrite<Expr>(op->priority_queue);
        }
	
	void MIRRewriter::visit(UpdatePriorityEdgeCountEdgeSetApplyExpr::Ptr ptr) {
	    //visit(std::static_pointer_cast<EdgeSetApplyExpr>(ptr));
            ptr->target = rewrite<Expr>(ptr->target);
            node = ptr;
	}

    }
}
