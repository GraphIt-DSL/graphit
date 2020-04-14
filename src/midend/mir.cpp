//
// Created by Yunming Zhang on 2/8/17.
//

#include <graphit/midend/mir.h>

namespace graphit {
    namespace mir {
        void Expr::copy(MIRNode::Ptr) {}

        MIRNode::Ptr Expr::cloneNode() {
            auto node = std::make_shared<mir::Expr>();
            return node;
        }

        void VarExpr::copy(MIRNode::Ptr node) {
            auto var_expr = to<mir::VarExpr>(node);
            var = Var(var_expr->var.getName(), var_expr->var.getType());
        }

        MIRNode::Ptr VarExpr::cloneNode() {
            const auto node = std::make_shared<VarExpr>();
            node->copy(shared_from_this());
            return node;
        }


        void TensorReadExpr::copy(MIRNode::Ptr node) {
            auto expr = to<mir::TensorReadExpr>(node);
            index = expr->index->clone<Expr>();
            target = expr->target->clone<Expr>();
            field_vector_prop_ = expr->field_vector_prop_;
        }


        MIRNode::Ptr TensorReadExpr::cloneNode() {
            const auto node = std::make_shared<TensorReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void TensorStructReadExpr::copy(MIRNode::Ptr node) {
            auto expr = to<mir::TensorStructReadExpr>(node);
            TensorReadExpr::copy(node);
            field_target = expr->field_target;
            array_of_struct_target = expr->array_of_struct_target;
        }


        MIRNode::Ptr TensorStructReadExpr::cloneNode() {
            const auto node = std::make_shared<TensorStructReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void TensorArrayReadExpr::copy(MIRNode::Ptr node) {
            auto expr = to<mir::TensorArrayReadExpr>(node);
            TensorReadExpr::copy(node);
        }


        MIRNode::Ptr TensorArrayReadExpr::cloneNode() {
            const auto node = std::make_shared<TensorArrayReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void Call::copy(MIRNode::Ptr node) {
            auto expr = to<mir::Call>(node);
            name = expr->name;
            for (const auto &arg: expr->args) {
                args.push_back(arg);
            }

            for (const auto &arg: expr->functorArgs) {
                functorArgs.push_back(arg);
            }

            if (generic_type != nullptr){
                generic_type = expr->generic_type->clone<Type>();
            }
        }


        MIRNode::Ptr Call::cloneNode() {
            const auto node = std::make_shared<Call>();
            node->copy(shared_from_this());
            return node;
        }


        void PriorityUpdateOperator::copy(MIRNode::Ptr node) {
            auto expr = to<mir::PriorityUpdateOperator>(node);
            Call::copy(node);
            destination_node_id = expr->destination_node_id;
            priority_queue = expr->priority_queue;
        }


        MIRNode::Ptr PriorityUpdateOperator::cloneNode() {
            const auto node = std::make_shared<PriorityUpdateOperator>();
            node->copy(shared_from_this());
            return node;
        }

        void PriorityUpdateOperatorMin::copy(MIRNode::Ptr node) {
            auto expr = to<mir::PriorityUpdateOperatorMin>(node);
            PriorityUpdateOperator::copy(node);
            new_val = expr->new_val;
            old_val = expr->old_val;
        }


        MIRNode::Ptr PriorityUpdateOperatorMin::cloneNode() {
            const auto node = std::make_shared<PriorityUpdateOperatorMin>();
            node->copy(shared_from_this());
            return node;
        }

        void PriorityUpdateOperatorSum::copy(MIRNode::Ptr node) {
            auto expr = to<mir::PriorityUpdateOperatorSum>(node);
            PriorityUpdateOperator::copy(node);
            delta = expr->delta;
            minimum_val = expr->minimum_val;
        }


        MIRNode::Ptr PriorityUpdateOperatorSum::cloneNode() {
            const auto node = std::make_shared<PriorityUpdateOperatorSum>();
            node->copy(shared_from_this());
            return node;
        }



        void LoadExpr::copy(MIRNode::Ptr node) {
            Expr::copy(node);
            auto expr = to<mir::LoadExpr>(node);
            file_name = expr->file_name->clone<Expr>();
        }


        MIRNode::Ptr LoadExpr::cloneNode() {
            const auto node = std::make_shared<LoadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void IntersectionExpr::copy(MIRNode::Ptr node) {
            auto expr = to<mir::IntersectionExpr>(node);
            vertex_a = expr->vertex_a->clone<Expr>();
            vertex_b = expr->vertex_b->clone<Expr>();
            numA = expr->numA->clone<Expr>();
            numB = expr->numB->clone<Expr>();
            if (expr->reference != nullptr){
                reference = expr->reference->clone<Expr>();
            }
            intersectionType = expr->intersectionType;

        }


        MIRNode::Ptr IntersectionExpr::cloneNode() {
            const auto node = std::make_shared<IntersectionExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void IntersectNeighborExpr::copy(MIRNode::Ptr node) {
            auto expr = to<mir::IntersectNeighborExpr>(node);
            vertex_a = expr->vertex_a->clone<Expr>();
            vertex_b = expr->vertex_b->clone<Expr>();
            intersectionType = expr->intersectionType;

        }


        MIRNode::Ptr IntersectNeighborExpr::cloneNode() {
            const auto node = std::make_shared<IntersectNeighborExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void EdgeSetLoadExpr::copy(MIRNode::Ptr node) {
            Expr::copy(node);
            auto expr = to<mir::EdgeSetLoadExpr>(node);
            file_name = expr->file_name->clone<Expr>();
            is_weighted_ = expr->is_weighted_;
        }


        MIRNode::Ptr EdgeSetLoadExpr::cloneNode() {
            const auto node = std::make_shared<EdgeSetLoadExpr>();
            node->copy(shared_from_this());
            return node;
        }


        void ApplyExpr::copy(MIRNode::Ptr node) {
            Expr::copy(node);
            auto expr = to<mir::ApplyExpr>(node);
            target = expr->target->clone<Expr>();
            input_function = expr->input_function->clone<FuncExpr>();
            tracking_field = expr->tracking_field;
        }


        MIRNode::Ptr ApplyExpr::cloneNode() {
            const auto node = std::make_shared<ApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void VertexSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = to<VertexSetApplyExpr>(node);
            ApplyExpr::copy(expr);
        }


        MIRNode::Ptr VertexSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<VertexSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void EdgeSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = to<EdgeSetApplyExpr>(node);
            ApplyExpr::copy(expr);

            if (expr->from_func) {
                from_func = expr->from_func->clone<FuncExpr>();
            }

            if (expr->to_func) {
                to_func = expr->to_func->clone<FuncExpr>();
            }

            is_parallel = expr->is_parallel;
            enable_deduplication = expr->enable_deduplication;
            is_weighted = expr->is_weighted;
            scope_label_name = expr->scope_label_name;
        }


        MIRNode::Ptr EdgeSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<EdgeSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void PushEdgeSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<PushEdgeSetApplyExpr>(node);
            EdgeSetApplyExpr::copy(expr);
        }


        MIRNode::Ptr PushEdgeSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<PushEdgeSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void PullEdgeSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<PullEdgeSetApplyExpr>(node);
            EdgeSetApplyExpr::copy(expr);
        }


        MIRNode::Ptr PullEdgeSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<PullEdgeSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void HybridDenseEdgeSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<HybridDenseEdgeSetApplyExpr>(node);
            EdgeSetApplyExpr::copy(expr);
            push_function_ = expr->push_function_;
        }


        MIRNode::Ptr HybridDenseEdgeSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<HybridDenseEdgeSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }


        void HybridDenseForwardEdgeSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<HybridDenseForwardEdgeSetApplyExpr>(node);
            EdgeSetApplyExpr::copy(node);
        }


        MIRNode::Ptr HybridDenseForwardEdgeSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<HybridDenseForwardEdgeSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }


        void WhereExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<WhereExpr>(node);
            target = expr->target;
            input_func = expr->input_func->clone<FuncExpr>();
            is_constant_set = expr->is_constant_set;
        }


        MIRNode::Ptr WhereExpr::cloneNode() {
            const auto node = std::make_shared<WhereExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void VertexSetWhereExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<VertexSetWhereExpr>(node);
            WhereExpr::copy(expr);
        }


        MIRNode::Ptr VertexSetWhereExpr::cloneNode() {
            const auto node = std::make_shared<VertexSetWhereExpr>();
            node->copy(shared_from_this());
            return node;
        }


        void EdgeSetWhereExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<EdgeSetWhereExpr>(node);
            WhereExpr::copy(expr);
        }


        MIRNode::Ptr EdgeSetWhereExpr::cloneNode() {
            const auto node = std::make_shared<EdgeSetWhereExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void VertexSetAllocExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<VertexSetAllocExpr>(node);
            size_expr = expr->size_expr->clone<Expr>();
            layout = expr->layout;
            element_type = expr->element_type;
        }

        MIRNode::Ptr VertexSetAllocExpr::cloneNode() {
            const auto node = std::make_shared<VertexSetAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }


        void VectorAllocExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<VectorAllocExpr>(node);
            size_expr = expr->size_expr->clone<Expr>();
            element_type = expr->element_type;
            if (expr->scalar_type != nullptr)
                scalar_type = expr->scalar_type->clone<ScalarType>();
            if (expr->vector_type !=nullptr)
                vector_type = expr->vector_type->clone<VectorType>();
        }

        MIRNode::Ptr VectorAllocExpr::cloneNode() {
            const auto node = std::make_shared<VectorAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void ListAllocExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<ListAllocExpr>(node);
            size_expr = expr->size_expr->clone<Expr>();
            element_type = expr->element_type;
        }

        MIRNode::Ptr ListAllocExpr::cloneNode() {
            const auto node = std::make_shared<ListAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void NaryExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<NaryExpr>(node);
            for (const auto &operand : expr->operands) {
                operands.push_back(operand);
            }
        }

        MIRNode::Ptr NaryExpr::cloneNode() {
            const auto node = std::make_shared<NaryExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void BinaryExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<BinaryExpr>(node);
            lhs = expr->lhs->clone<Expr>();
            rhs = expr->rhs->clone<Expr>();
        }

        MIRNode::Ptr BinaryExpr::cloneNode() {
            const auto node = std::make_shared<BinaryExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void NegExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<NegExpr>(node);
            operand = expr->operand->clone<Expr>();
            negate = expr->negate;
        }

        MIRNode::Ptr NegExpr::cloneNode() {
            const auto node = std::make_shared<NegExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void EqExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<EqExpr>(node);
            NaryExpr::copy(expr);
            for (const auto &op : expr->ops) {
                ops.push_back(op);
            }
        }

        MIRNode::Ptr EqExpr::cloneNode() {
            const auto node = std::make_shared<EqExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void AndExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<AndExpr>(node);
            BinaryExpr::copy(expr);
        }

        MIRNode::Ptr AndExpr::cloneNode() {
            const auto node = std::make_shared<AndExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void OrExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<OrExpr>(node);
            BinaryExpr::copy(expr);
        }

        MIRNode::Ptr OrExpr::cloneNode() {
            const auto node = std::make_shared<OrExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void XorExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<XorExpr>(node);
            BinaryExpr::copy(expr);
        }

        MIRNode::Ptr XorExpr::cloneNode() {
            const auto node = std::make_shared<XorExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void NotExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<NotExpr>(node);
            operand = expr->operand->clone<Expr>();
        }

        MIRNode::Ptr NotExpr::cloneNode() {
            const auto node = std::make_shared<NotExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void AddExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<AddExpr>(node);
            BinaryExpr::copy(expr);
        }

        MIRNode::Ptr AddExpr::cloneNode() {
            const auto node = std::make_shared<AddExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void MulExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<MulExpr>(node);
            BinaryExpr::copy(expr);
        }

        MIRNode::Ptr MulExpr::cloneNode() {
            const auto node = std::make_shared<MulExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void DivExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<DivExpr>(node);
            BinaryExpr::copy(expr);
        }

        MIRNode::Ptr DivExpr::cloneNode() {
            const auto node = std::make_shared<DivExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void SubExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<SubExpr>(node);
            BinaryExpr::copy(expr);
        }

        MIRNode::Ptr SubExpr::cloneNode() {
            const auto node = std::make_shared<SubExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void StringLiteral::copy(MIRNode::Ptr node) {
            auto str_lit = to<mir::StringLiteral>(node);
            val = str_lit->val;
        }


        MIRNode::Ptr StringLiteral::cloneNode() {
            const auto node = std::make_shared<StringLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void IntLiteral::copy(MIRNode::Ptr node) {
            auto lit = to<mir::IntLiteral>(node);
            val = lit->val;
        }


        MIRNode::Ptr IntLiteral::cloneNode() {
            const auto node = std::make_shared<IntLiteral>();
            node->copy(shared_from_this());
            return node;
        }


        void BoolLiteral::copy(MIRNode::Ptr node) {
            auto lit = to<mir::BoolLiteral>(node);
            val = lit->val;
        }


        MIRNode::Ptr BoolLiteral::cloneNode() {
            const auto node = std::make_shared<BoolLiteral>();
            node->copy(shared_from_this());
            return node;
        }


        void FloatLiteral::copy(MIRNode::Ptr node) {
            auto lit = to<mir::FloatLiteral>(node);
            val = lit->val;
        }


        MIRNode::Ptr FloatLiteral::cloneNode() {
            const auto node = std::make_shared<FloatLiteral>();
            node->copy(shared_from_this());
            return node;
        }


        void Stmt::copy(MIRNode::Ptr node) {
            auto stmt = to<mir::Stmt>(node);
        }


        MIRNode::Ptr Stmt::cloneNode() {
            const auto node = std::make_shared<Stmt>();
            node->copy(shared_from_this());
            return node;
        }

        void StmtBlock::copy(MIRNode::Ptr node) {
            auto stmt_blk = to<mir::StmtBlock>(node);
            stmts = new std::vector<Stmt::Ptr>();
            for (auto &stmt : (*(stmt_blk->stmts))) {
                stmts->push_back(stmt->clone<Stmt>());
            }
        }


        MIRNode::Ptr StmtBlock::cloneNode() {
            const auto node = std::make_shared<StmtBlock>();
            node->copy(shared_from_this());
            return node;
        }

        void ScalarType::copy(MIRNode::Ptr node) {
            auto type_node = to<mir::ScalarType>(node);
            type = type_node->type;
        }


        MIRNode::Ptr ScalarType::cloneNode() {
            const auto node = std::make_shared<ScalarType>();
            node->copy(shared_from_this());
            return node;
        }

        void ElementType::copy(MIRNode::Ptr node) {
            auto type_node = to<mir::ElementType>(node);
            ident = type_node->ident;
        }


        MIRNode::Ptr ElementType::cloneNode() {
            const auto node = std::make_shared<ElementType>();
            node->copy(shared_from_this());
            return node;
        }

        void VectorType::copy(MIRNode::Ptr node) {
            auto type_node = to<mir::VectorType>(node);
            element_type = type_node->element_type->clone<ElementType>();
            vector_element_type = type_node->vector_element_type->clone<Type>();
        }


        MIRNode::Ptr VectorType::cloneNode() {
            const auto node = std::make_shared<VectorType>();
            node->copy(shared_from_this());
            return node;
        }


        void VertexSetType::copy(MIRNode::Ptr node) {
            auto type_node = to<mir::VertexSetType>(node);
            element = type_node->element->clone<ElementType>();
        }


        MIRNode::Ptr VertexSetType::cloneNode() {
            const auto node = std::make_shared<VertexSetType>();
            node->copy(shared_from_this());
            return node;
        }

        void ListType::copy(MIRNode::Ptr node) {
            auto type_node = to<mir::ListType>(node);
            element_type = type_node->element_type->clone<ListType>();
        }


        MIRNode::Ptr ListType::cloneNode() {
            const auto node = std::make_shared<ListType>();
            node->copy(shared_from_this());
            return node;
        }

        void EdgeSetType::copy(MIRNode::Ptr node) {
            auto type_node = to<mir::EdgeSetType>(node);
            element = type_node->element->clone<ElementType>();
            weight_type = type_node->weight_type->clone<ScalarType>();
        }


        MIRNode::Ptr EdgeSetType::cloneNode() {
            const auto node = std::make_shared<EdgeSetType>();
            node->copy(shared_from_this());
            return node;
        }

        void ForDomain::copy(MIRNode::Ptr node) {
            auto for_domain_node = to<mir::ForDomain>(node);
            lower = for_domain_node->lower->clone<Expr>();
            upper = for_domain_node->upper->clone<Expr>();
        }


        MIRNode::Ptr ForDomain::cloneNode() {
            const auto node = std::make_shared<ForDomain>();
            node->copy(shared_from_this());
            return node;
        }

        void NameNode::copy(MIRNode::Ptr node) {
            auto name_node = to<mir::NameNode>(node);
            body = name_node->body->clone<StmtBlock>();
        }


        MIRNode::Ptr NameNode::cloneNode() {
            const auto node = std::make_shared<NameNode>();
            node->copy(shared_from_this());
            return node;
        }

        void ForStmt::copy(MIRNode::Ptr node) {
            auto for_node = to<mir::ForStmt>(node);
            loopVar = for_node->loopVar;
            domain = for_node->domain->clone<ForDomain>();
            body = for_node->body->clone<StmtBlock>();
        }


        MIRNode::Ptr ForStmt::cloneNode() {
            const auto node = std::make_shared<ForStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void WhileStmt::copy(MIRNode::Ptr node) {
            auto while_stmt = to<mir::WhileStmt>(node);
            cond = while_stmt->cond->clone<Expr>();
            body = while_stmt->body->clone<StmtBlock>();
        }


        MIRNode::Ptr WhileStmt::cloneNode() {
            const auto node = std::make_shared<WhileStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void ExprStmt::copy(MIRNode::Ptr node) {
            auto expr_stmt = to<mir::ExprStmt>(node);
            expr = expr_stmt->expr->clone<Expr>();
        }


        MIRNode::Ptr ExprStmt::cloneNode() {
            const auto node = std::make_shared<ExprStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void AssignStmt::copy(MIRNode::Ptr node) {
            auto expr_stmt = to<mir::AssignStmt>(node);
            expr = expr_stmt->expr->clone<Expr>();
            lhs = expr_stmt->lhs->clone<Expr>();
        }


        MIRNode::Ptr AssignStmt::cloneNode() {
            const auto node = std::make_shared<AssignStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void ReduceStmt::copy(MIRNode::Ptr node) {
            auto stmt = to<mir::ReduceStmt>(node);
            expr = stmt->expr->clone<Expr>();
            lhs = stmt->lhs->clone<Expr>();
            reduce_op_ = stmt->reduce_op_;
            tracking_var_name_ = stmt->tracking_var_name_;
            is_atomic_ = stmt->is_atomic_;
        }


        MIRNode::Ptr ReduceStmt::cloneNode() {
            const auto node = std::make_shared<ReduceStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void CompareAndSwapStmt::copy(MIRNode::Ptr node) {
            auto stmt = to<mir::CompareAndSwapStmt>(node);
            compare_val_expr = stmt->compare_val_expr->clone<Expr>();
            tracking_var_ = stmt->tracking_var_;
        }


        MIRNode::Ptr CompareAndSwapStmt::cloneNode() {
            const auto node = std::make_shared<CompareAndSwapStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void PrintStmt::copy(MIRNode::Ptr node) {
            auto stmt = to<mir::PrintStmt>(node);
            expr = stmt->expr->clone<Expr>();
            format = stmt->format;
        }


        MIRNode::Ptr PrintStmt::cloneNode() {
            const auto node = std::make_shared<PrintStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void BreakStmt::copy(MIRNode::Ptr node) {
            auto stmt = to<mir::BreakStmt>(node);
        }


        MIRNode::Ptr BreakStmt::cloneNode() {
            const auto node = std::make_shared<BreakStmt>();
            node->copy(shared_from_this());
            return node;
        }


        void IfStmt::copy(MIRNode::Ptr node) {
            auto stmt = to<mir::IfStmt>(node);
            cond = stmt->cond->clone<Expr>();
            ifBody = stmt->ifBody->clone<Stmt>();
            if (stmt->elseBody != nullptr)
                elseBody = stmt->elseBody->clone<Stmt>();

        }

        MIRNode::Ptr IfStmt::cloneNode() {
            const auto node = std::make_shared<IfStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void IdentDecl::copy(MIRNode::Ptr node) {
            auto decl = to<IdentDecl>(node);
            if (decl->type) {
                type = decl->type->clone<Type>();
            }
            name = decl->name;
        }


        MIRNode::Ptr IdentDecl::cloneNode() {
            const auto node = std::make_shared<IdentDecl>();
            node->copy(shared_from_this());
            return node;
        }


        void VarDecl::copy(MIRNode::Ptr node) {
            auto decl = to<VarDecl>(node);
            type = decl->type->clone<Type>();
            initVal = decl->initVal->clone<Expr>();
            modifier = decl->modifier;
            name = decl->name;
        }


        MIRNode::Ptr VarDecl::cloneNode() {
            const auto node = std::make_shared<VarDecl>();
            node->copy(shared_from_this());
            return node;
        }

        void StructTypeDecl::copy(MIRNode::Ptr node) {
            auto decl = to<StructTypeDecl>(node);
            name = decl->name;
            for (const auto &field : decl->fields) {
                fields.push_back(field);
            }
        }


        MIRNode::Ptr StructTypeDecl::cloneNode() {
            const auto node = std::make_shared<StructTypeDecl>();
            node->copy(shared_from_this());
            return node;
        }

        void FuncExpr::copy(MIRNode::Ptr node) {

            auto funcExpr = to<FuncExpr>(node);

            for(auto &arg : funcExpr->functorArgs) {
                functorArgs.push_back(arg);
            }

            function_name = funcExpr->function_name->clone<IdentDecl>();

        }

        MIRNode::Ptr FuncExpr::cloneNode() {
            const auto node = std::make_shared<FuncExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void FuncDecl::copy(MIRNode::Ptr node) {
            auto decl = to<FuncDecl>(node);
            name = decl->name;
            for (const auto &arg : decl->args) {
                args.push_back(arg);
            }

            for (const auto &arg : decl->functorArgs) {
                functorArgs.push_back(arg);
            }
            body = decl->body->clone<StmtBlock>();
            if (decl->result.isInitialized())
                result = mir::Var(decl->result.getName(), decl->result.getType());
            //I am not sure, I think this just copies over everything
            field_vector_properties_map_ = decl->field_vector_properties_map_;
        }


        MIRNode::Ptr FuncDecl::cloneNode() {
            const auto node = std::make_shared<FuncDecl>();
            node->copy(shared_from_this());
            return node;
        }

        void PriorityQueueType::copy(MIRNode::Ptr node) {
            auto type_node = to<mir::PriorityQueueType>(node);
            element = type_node->element->clone<ElementType>();
        }


        MIRNode::Ptr PriorityQueueType::cloneNode() {
            const auto node = std::make_shared<PriorityQueueType>();
            node->copy(shared_from_this());
            return node;
        }

        void PriorityQueueAllocExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<PriorityQueueAllocExpr>(node);
            element_type = expr->element_type;
            dup_within_bucket = expr->dup_within_bucket;
            dup_across_bucket = expr->dup_across_bucket;
            vector_function = expr->vector_function;
            bucket_ordering = expr->bucket_ordering;
            priority_ordering = expr->priority_ordering;
            init_bucket = expr->init_bucket;
            starting_node = expr->starting_node;

        }

        MIRNode::Ptr PriorityQueueAllocExpr::cloneNode() {
            const auto node = std::make_shared<PriorityQueueAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void UpdatePriorityEdgeSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<UpdatePriorityEdgeSetApplyExpr>(node);
            EdgeSetApplyExpr::copy(expr);
        }


        MIRNode::Ptr UpdatePriorityEdgeSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<UpdatePriorityEdgeSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void UpdatePriorityExternVertexSetApplyExpr::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<UpdatePriorityExternVertexSetApplyExpr>(node);
            VertexSetApplyExpr::copy(expr);
        }


        MIRNode::Ptr UpdatePriorityExternVertexSetApplyExpr::cloneNode() {
            const auto node = std::make_shared<UpdatePriorityExternVertexSetApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void UpdatePriorityUpdateBucketsCall::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<UpdatePriorityUpdateBucketsCall>(node);
            priority_queue_name = expr->priority_queue_name;
            lambda_name = expr->lambda_name;
            modified_vertexsubset_name = expr->modified_vertexsubset_name;
            nodes_init_in_bucket = expr->nodes_init_in_bucket;
        }

        MIRNode::Ptr UpdatePriorityUpdateBucketsCall::cloneNode() {
            const auto node = std::make_shared<UpdatePriorityUpdateBucketsCall>();
            node->copy(shared_from_this());
            return node;
        }

        void UpdatePriorityExternCall::copy(MIRNode::Ptr node) {
            const auto expr = mir::to<UpdatePriorityExternCall>(node);
            input_set = expr->input_set;
            priority_queue_name = expr->priority_queue_name;
            output_set_name = expr->output_set_name;
            lambda_name = expr->lambda_name;
            apply_function_name = expr->apply_function_name;
        }

        MIRNode::Ptr UpdatePriorityExternCall::cloneNode() {
            const auto node = std::make_shared<UpdatePriorityExternCall>();
            node->copy(shared_from_this());
            return node;
        }

        void OrderedProcessingOperator::copy(MIRNode::Ptr node) {
            const auto op = mir::to<OrderedProcessingOperator>(node);
            edge_update_func = op->edge_update_func->clone<FuncExpr>();
            while_cond_expr = op->while_cond_expr;
            optional_source_node = op->optional_source_node;
            priority_queue_name = op->priority_queue_name;
            priority_udpate_type = op->priority_udpate_type;
            merge_threshold = op->merge_threshold;
        }

        MIRNode::Ptr OrderedProcessingOperator::cloneNode() {
            const auto node = std::make_shared<OrderedProcessingOperator>();
            node->copy(shared_from_this());
            return node;
        }

	void UpdatePriorityEdgeCountEdgeSetApplyExpr::copy(MIRNode::Ptr node) {
		const auto op = mir::to<UpdatePriorityEdgeCountEdgeSetApplyExpr>(node);
		lambda_name = op->lambda_name;
		moved_object_name = op->moved_object_name;
		EdgeSetApplyExpr::copy(node);	
	}
	MIRNode::Ptr UpdatePriorityEdgeCountEdgeSetApplyExpr::cloneNode() {
		const auto node = std::make_shared<UpdatePriorityEdgeCountEdgeSetApplyExpr>();
		node->copy(shared_from_this());
		return node;
	}

    }
}
