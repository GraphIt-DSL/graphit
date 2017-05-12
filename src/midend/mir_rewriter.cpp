//
// Created by Yunming Zhang on 5/12/17.
//

#include <graphit/midend/mir_rewriter.h>


namespace graphit {
    namespace mir {

        void MIRRewriter::visit(Expr::Ptr expr) {
            expr->accept(this);
        };


        void MIRRewriter::visit(ExprStmt::Ptr stmt) {
            stmt->expr->accept(this);
        }

        void MIRRewriter::visit(AssignStmt::Ptr stmt) {
            stmt->lhs->accept(this);
            stmt->expr->accept(this);
        }

        void MIRRewriter::visit(PrintStmt::Ptr stmt) {
            stmt->expr->accept(this);
        }

        void MIRRewriter::visit(StmtBlock::Ptr stmt_block) {
            for (auto stmt : *(stmt_block->stmts)) {
                stmt->accept(this);
            }
        }

        void MIRRewriter::visit(FuncDecl::Ptr func_decl) {

//            for (auto arg : func_decl->args) {
//                arg->accept(this);
//            }
//            func_decl->result->accept(this);

            if (func_decl->body->stmts) {
                func_decl->body->accept(this);
            }
        }

        void MIRRewriter::visit(Call::Ptr expr) {
            for (auto arg : expr->args){
                arg->accept(this);
            }
        };
        void MIRRewriter::visit(MulExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(DivExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(AddExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(SubExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRRewriter::visit(std::shared_ptr<VarDecl> var_decl) {
            var_decl->initVal->accept(this);
        }



        void MIRRewriter::visit(std::shared_ptr<VertexSetAllocExpr> expr) {
            expr->size_expr->accept(this);
        }

        void MIRRewriter::visit(std::shared_ptr<ApplyExpr> expr) {
            expr->target->accept(this);
        }

        void MIRRewriter::visit(std::shared_ptr<TensorReadExpr> expr) {
            expr->target->accept(this);
            expr->index->accept(this);
        }

        void MIRRewriter::visit(std::shared_ptr<ForStmt> for_stmt) {
            for_stmt->domain->accept(this);
            for_stmt->body->accept(this);
        }

        void MIRRewriter::visit(std::shared_ptr<ForDomain> for_domain) {
            for_domain->lower->accept(this);
            for_domain->upper->accept(this);
        }

        void MIRRewriter::visit(std::shared_ptr<StructTypeDecl> struct_type_decl) {
            for (auto field : struct_type_decl->fields) {
                field->accept(this);
            }
        }

        void MIRRewriter::visit(std::shared_ptr<VertexSetType> vertexset_type) {
            vertexset_type->element->accept(this);
        }

        void MIRRewriter::visit(std::shared_ptr<EdgeSetType> edgeset_type) {
            edgeset_type->element->accept(this);
            for (auto element_type : *edgeset_type->vertex_element_type_list){
                element_type->accept(this);
            }
        }

        void MIRRewriter::visit(std::shared_ptr<VectorType> vector_type) {
            vector_type->element_type->accept(this);
            vector_type->vector_element_type->accept(this);
        }

        void MIRRewriter::visitBinaryExpr(BinaryExpr::Ptr expr) {
            expr->lhs = rewrite<Expr> (expr->lhs);
            expr->rhs = rewrite<Expr> (expr->rhs);
            node = expr;
        }

    }
}