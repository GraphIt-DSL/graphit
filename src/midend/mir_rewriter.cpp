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
            for (auto stmt : *(stmt_block->stmts)) {
                stmt = rewrite<Stmt>(stmt);
            }
            node = stmt_block;
        }


        void MIRRewriter::visit(FuncDecl::Ptr func_decl) {

            if (func_decl->body->stmts) {
                func_decl->body = rewrite<StmtBlock>(func_decl->body);
            }
            node = func_decl;
        }

        void MIRRewriter::visit(Call::Ptr expr) {
            // need to use & to actually modify the argument (with reference), not a copy of it
            for (auto &arg : expr->args) {
                arg = rewrite<Expr>(arg);
            }
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
            if (var_decl->initVal != nullptr)
	            var_decl->initVal = rewrite<Expr>(var_decl->initVal);
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

    }
}
