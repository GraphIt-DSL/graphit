//
// Created by Yunming Zhang on 2/10/17.
//

//
// Created by Yunming Zhang on 1/24/17.
//


#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir.h>

namespace graphit {
    namespace mir {

        void MIRVisitor::visit(Expr::Ptr expr) {
            expr->accept(this);
        };


        void MIRVisitor::visit(ExprStmt::Ptr stmt) {
            stmt->expr->accept(this);
        }

        void MIRVisitor::visit(AssignStmt::Ptr stmt) {
            stmt->lhs->accept(this);
            stmt->expr->accept(this);
        }

        void MIRVisitor::visit(PrintStmt::Ptr stmt) {
            stmt->expr->accept(this);
        }

        void MIRVisitor::visit(StmtBlock::Ptr stmt_block) {
            for (auto stmt : *(stmt_block->stmts)) {
                stmt->accept(this);
            }
        }

        void MIRVisitor::visit(FuncDecl::Ptr func_decl) {

//            for (auto arg : func_decl->args) {
//                arg->accept(this);
//            }
//            func_decl->result->accept(this);

            if (func_decl->body->stmts) {
                func_decl->body->accept(this);
            }
        }

        void MIRVisitor::visit(Call::Ptr expr) {
            for (auto arg : expr->args){
                arg->accept(this);
            }
        };
        void MIRVisitor::visit(MulExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRVisitor::visit(DivExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRVisitor::visit(EqExpr::Ptr expr) {
            visitNaryExpr(expr);
        }

        void MIRVisitor::visit(AddExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRVisitor::visit(SubExpr::Ptr expr) {
            visitBinaryExpr(expr);
        }

        void MIRVisitor::visit(std::shared_ptr<VarDecl> var_decl) {
            var_decl->initVal->accept(this);
        }

        void MIRVisitor::visitBinaryExpr(BinaryExpr::Ptr expr) {
            expr->lhs->accept(this);
            expr->rhs->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<VertexSetAllocExpr> expr) {
            expr->size_expr->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<VertexSetApplyExpr> expr) {
            expr->target->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<EdgeSetApplyExpr> expr) {
            expr->target->accept(this);
            expr->from_expr->accept(this);
            expr->to_expr->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<VertexSetWhereExpr> expr) {
            //expr->target->accept(this);
            //expr->input_func->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<EdgeSetWhereExpr> expr) {
            //expr->target->accept(this);
            //expr->input_func->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<TensorReadExpr> expr) {
            expr->target->accept(this);
            expr->index->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<TensorArrayReadExpr> expr) {
            expr->target->accept(this);
            expr->index->accept(this);

        }

        void MIRVisitor::visit(std::shared_ptr<TensorStructReadExpr> expr) {
            expr->index->accept(this);
            expr->field_target->accept(this);
            // This is now changed to a string
            //expr->struct_target->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<ForStmt> for_stmt) {
            for_stmt->domain->accept(this);
            for_stmt->body->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<ForDomain> for_domain) {
            for_domain->lower->accept(this);
            for_domain->upper->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<StructTypeDecl> struct_type_decl) {
            for (auto field : struct_type_decl->fields) {
                field->accept(this);
            }
        }

        void MIRVisitor::visit(std::shared_ptr<VertexSetType> vertexset_type) {
            vertexset_type->element->accept(this);
        }

        void MIRVisitor::visit(std::shared_ptr<EdgeSetType> edgeset_type) {
            edgeset_type->element->accept(this);
            for (auto element_type : *edgeset_type->vertex_element_type_list){
                element_type->accept(this);
            }
        }

        void MIRVisitor::visit(std::shared_ptr<VectorType> vector_type) {
            vector_type->element_type->accept(this);
            vector_type->vector_element_type->accept(this);
        }


        void MIRVisitor::visitNaryExpr(NaryExpr::Ptr expr) {
            for (auto operand : expr->operands) {
                operand->accept(this);
            }
        }

    }
}