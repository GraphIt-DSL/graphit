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
            stmt->expr = rewrite<Expr>(stmt->expr);
            node = stmt;
        }

        void MIRRewriter::visit(AssignStmt::Ptr stmt) {
            stmt->lhs = rewrite<Expr>(stmt->lhs);
            stmt->expr = rewrite<Expr>(stmt->expr);
            node = stmt;
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
            for (auto arg : expr->args){
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
            var_decl->initVal = rewrite<Expr>(var_decl->initVal);
            node = var_decl;
        }


        void MIRRewriter::visit(std::shared_ptr<VertexSetAllocExpr> expr) {
            expr->size_expr = rewrite<Expr>(expr->size_expr);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<VertexSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            node = expr;
        }

        void MIRRewriter::visit(std::shared_ptr<EdgeSetApplyExpr> expr) {
            expr->target = rewrite<Expr>(expr->target);
            //if(expr->from_func) expr->from_func = rewrite<Expr>(expr->from_func);
            //if(expr->to_func)   expr->to_func = rewrite<Expr>(expr->to_func);
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

        void MIRRewriter::visit(std::shared_ptr<ForStmt> for_stmt) {
            for_stmt->domain = rewrite<ForDomain>(for_stmt->domain);
            for_stmt->body = rewrite<StmtBlock>(for_stmt->body);
            node = for_stmt;
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

        void MIRRewriter::visit(std::shared_ptr<EdgeSetType> edgeset_type) {
            edgeset_type->element = rewrite<ElementType>(edgeset_type->element);
            for (auto element_type : *edgeset_type->vertex_element_type_list){
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
            expr->lhs = rewrite<Expr> (expr->lhs);
            expr->rhs = rewrite<Expr> (expr->rhs);
            node = expr;
        }

        void MIRRewriter::visitNaryExpr(NaryExpr::Ptr expr) {
            // Here we are modifying the original operand, so need the & before operand
            for (auto &operand : expr->operands) {
                operand = rewrite<Expr>(operand);
            }
            node = expr;
        }

    }
}