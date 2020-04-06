//
// Created by Yunming Zhang on 1/24/17.
//

#include <graphit/frontend/fir.h>
#include <graphit/frontend/token.h>

namespace graphit {
    namespace fir {

        void FIRNode::copy(FIRNode::Ptr node) {
            lineBegin = node->lineBegin;
            colBegin = node->colBegin;
            lineEnd = node->lineEnd;
            colEnd = node->colEnd;
        }

        void FIRNode::setBeginLoc(const Token &token) {
            lineBegin = token.lineBegin;
            colBegin = token.colBegin;
        }

        void FIRNode::setEndLoc(const Token &token) {
            lineEnd = token.lineEnd;
            colEnd = token.colEnd;
        }

        void FIRNode::setLoc(const Token &token) {
            setBeginLoc(token);
            setEndLoc(token);
        }

        void Program::copy(FIRNode::Ptr node) {
            const auto program = to<Program>(node);
            FIRNode::copy(program);
            for (const auto &elem : program->elems) {
                elems.push_back(elem->clone());
            }
        }

        FIRNode::Ptr Program::cloneNode() {
            const auto node = std::make_shared<Program>();
            node->copy(shared_from_this());
            return node;
        }

        void StmtBlock::copy(FIRNode::Ptr node) {
            const auto stmtBlock = to<StmtBlock>(node);
            Stmt::copy(stmtBlock);
            for (const auto &stmt : stmtBlock->stmts) {
                stmts.push_back(stmt->clone<Stmt>());
            }
        }

        FIRNode::Ptr StmtBlock::cloneNode() {
            const auto node = std::make_shared<StmtBlock>();
            node->copy(shared_from_this());
            return node;
        }

        void RangeIndexSet::copy(FIRNode::Ptr node) {
            const auto indexSet = to<RangeIndexSet>(node);
            IndexSet::copy(indexSet);
            range = indexSet->range;
        }

        FIRNode::Ptr RangeIndexSet::cloneNode() {
            const auto node = std::make_shared<RangeIndexSet>();
            node->copy(shared_from_this());
            return node;
        }

        void SetIndexSet::copy(FIRNode::Ptr node) {
            const auto indexSet = to<SetIndexSet>(node);
            IndexSet::copy(indexSet);
            setName = indexSet->setName;
        }

        FIRNode::Ptr SetIndexSet::cloneNode() {
            const auto node = std::make_shared<SetIndexSet>();
            node->copy(shared_from_this());
            return node;
        }

        void GenericIndexSet::copy(FIRNode::Ptr node) {
            const auto indexSet = to<GenericIndexSet>(node);
            SetIndexSet::copy(indexSet);
            type = indexSet->type;
        }

        FIRNode::Ptr GenericIndexSet::cloneNode() {
            const auto node = std::make_shared<GenericIndexSet>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr DynamicIndexSet::cloneNode() {
            const auto node = std::make_shared<DynamicIndexSet>();
            node->copy(shared_from_this());
            return node;
        }

        void ElementType::copy(FIRNode::Ptr node) {
            const auto elementType = to<ElementType>(node);
            Type::copy(elementType);
            ident = elementType->ident;
        }

        FIRNode::Ptr ElementType::cloneNode() {
            const auto node = std::make_shared<ElementType>();
            node->copy(shared_from_this());
            return node;
        }

        void Endpoint::copy(FIRNode::Ptr node) {
            const auto endpoint = to<Endpoint>(node);
            FIRNode::copy(endpoint);
            set = endpoint->set->clone<SetIndexSet>();
            element = endpoint->element->clone<ElementType>();
        }

        FIRNode::Ptr Endpoint::cloneNode() {
            const auto node = std::make_shared<Endpoint>();
            node->copy(shared_from_this());
            return node;
        }

        SetType::Ptr SetType::getUndefinedSetType() {
            const auto undefinedSetType = std::make_shared<HeterogeneousEdgeSetType>();
            undefinedSetType->element = std::make_shared<ElementType>();
            return undefinedSetType;
        }

        void SetType::copy(FIRNode::Ptr node) {
            const auto setType = to<SetType>(node);
            Type::copy(setType);
            element = setType->element->clone<ElementType>();
        }

        void TupleLength::copy(FIRNode::Ptr node) {
            const auto length = to<TupleLength>(node);
            FIRNode::copy(length);
            val = length->val;
        }

        FIRNode::Ptr TupleLength::cloneNode() {
            const auto node = std::make_shared<TupleLength>();
            node->copy(shared_from_this());
            return node;
        }

        void HomogeneousEdgeSetType::copy(FIRNode::Ptr node) {
            const auto setType = to<HomogeneousEdgeSetType>(node);
            UnstructuredSetType::copy(setType);
            endpoint = setType->endpoint->clone<Endpoint>();
            arity = setType->arity->clone<TupleLength>();
        }

        FIRNode::Ptr HomogeneousEdgeSetType::cloneNode() {
            const auto node = std::make_shared<HomogeneousEdgeSetType>();
            node->copy(shared_from_this());
            return node;
        }

        bool HeterogeneousEdgeSetType::isHomogeneous() const {
            const auto neighborSet = endpoints[0]->set->setName;

            for (unsigned i = 1; i < endpoints.size(); ++i) {
                if (endpoints[i]->set->setName != neighborSet) {
                    return false;
                }
            }

            return true;
        }

        void HeterogeneousEdgeSetType::copy(FIRNode::Ptr node) {
            const auto setType = to<HeterogeneousEdgeSetType>(node);
            UnstructuredSetType::copy(setType);
            for (const auto &endpoint : setType->endpoints) {
                endpoints.push_back(endpoint->clone<Endpoint>());
            }
        }

        FIRNode::Ptr HeterogeneousEdgeSetType::cloneNode() {
            const auto node = std::make_shared<HeterogeneousEdgeSetType>();
            node->copy(shared_from_this());
            return node;
        }

        void GridSetType::copy(FIRNode::Ptr node) {
            const auto setType = to<GridSetType>(node);
            SetType::copy(setType);
            underlyingPointSet = setType->underlyingPointSet->clone<Endpoint>();
            dimensions = setType->dimensions;
        }

        FIRNode::Ptr GridSetType::cloneNode() {
            const auto node = std::make_shared<GridSetType>();
            node->copy(shared_from_this());
            return node;
        }

        void Identifier::copy(FIRNode::Ptr node) {
            const auto identifier = to<Identifier>(node);
            FIRNode::copy(identifier);
            ident = identifier->ident;
        }

        FIRNode::Ptr Identifier::cloneNode() {
            const auto node = std::make_shared<Identifier>();
            node->copy(shared_from_this());
            return node;
        }

        void TupleElement::copy(FIRNode::Ptr node) {
            const auto elem = to<TupleElement>(node);
            FIRNode::copy(elem);
            if (elem->name) {
                name = elem->name->clone<Identifier>();
            }
            element = elem->element->clone<ElementType>();
        }

        FIRNode::Ptr TupleElement::cloneNode() {
            const auto node = std::make_shared<TupleElement>();
            node->copy(shared_from_this());
            return node;
        }

        void NamedTupleType::copy(FIRNode::Ptr node) {
            const auto tupleType = to<NamedTupleType>(node);
            Type::copy(tupleType);
            for (const auto &elem : tupleType->elems) {
                elems.push_back(elem->clone<TupleElement>());
            }
        }

        FIRNode::Ptr NamedTupleType::cloneNode() {
            const auto node = std::make_shared<NamedTupleType>();
            node->copy(shared_from_this());
            return node;
        }

        void UnnamedTupleType::copy(FIRNode::Ptr node) {
            const auto tupleType = to<UnnamedTupleType>(node);
            Type::copy(tupleType);
            element = tupleType->element->clone<ElementType>();
            length = tupleType->length->clone<TupleLength>();
        }

        FIRNode::Ptr UnnamedTupleType::cloneNode() {
            const auto node = std::make_shared<UnnamedTupleType>();
            node->copy(shared_from_this());
            return node;
        }

        void ScalarType::copy(FIRNode::Ptr node) {
            const auto scalarType = to<ScalarType>(node);
            TensorType::copy(scalarType);
            type = scalarType->type;
        }

        FIRNode::Ptr ScalarType::cloneNode() {
            const auto node = std::make_shared<ScalarType>();
            node->copy(shared_from_this());
            return node;
        }

        void NDTensorType::copy(FIRNode::Ptr node) {
            const auto ndTensorType = to<NDTensorType>(node);
            TensorType::copy(ndTensorType);
            for (const auto &indexSet : ndTensorType->indexSets) {
                indexSets.push_back(indexSet->clone<IndexSet>());
            }
            blockType = ndTensorType->blockType->clone<TensorType>();
            transposed = ndTensorType->transposed;
        }

        FIRNode::Ptr NDTensorType::cloneNode() {
            const auto node = std::make_shared<NDTensorType>();
            node->copy(shared_from_this());
            return node;
        }

        void OpaqueType::copy(FIRNode::Ptr node) {
            const auto opaqueType = to<OpaqueType>(node);
            Type::copy(opaqueType);
        }

        FIRNode::Ptr OpaqueType::cloneNode() {
            const auto node = std::make_shared<OpaqueType>();
            node->copy(shared_from_this());
            return node;
        }

        void IdentDecl::copy(FIRNode::Ptr node) {
            const auto identDecl = to<IdentDecl>(node);
            FIRNode::copy(identDecl);
            name = identDecl->name->clone<Identifier>();
            type = identDecl->type->clone<Type>();
        }

        FIRNode::Ptr IdentDecl::cloneNode() {
            const auto node = std::make_shared<IdentDecl>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr FieldDecl::cloneNode() {
            const auto node = std::make_shared<FieldDecl>();
            node->copy(shared_from_this());
            return node;
        }

        void ElementTypeDecl::copy(FIRNode::Ptr node) {
            const auto elementTypeDecl = to<ElementTypeDecl>(node);
            FIRNode::copy(elementTypeDecl);
            name = elementTypeDecl->name->clone<Identifier>();
            for (const auto &field : elementTypeDecl->fields) {
                fields.push_back(field->clone<FieldDecl>());
            }
        }

        FIRNode::Ptr ElementTypeDecl::cloneNode() {
            const auto node = std::make_shared<ElementTypeDecl>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr Argument::cloneNode() {
            const auto node = std::make_shared<Argument>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr InOutArgument::cloneNode() {
            const auto node = std::make_shared<InOutArgument>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr ExternDecl::cloneNode() {
            const auto node = std::make_shared<ExternDecl>();
            node->copy(shared_from_this());
            return node;
        }

        void GenericParam::copy(FIRNode::Ptr node) {
            const auto genericParam = to<GenericParam>(node);
            FIRNode::copy(genericParam);
            name = genericParam->name;
            type = genericParam->type;
        }

        FIRNode::Ptr GenericParam::cloneNode() {
            const auto node = std::make_shared<GenericParam>();
            node->copy(shared_from_this());
            return node;
        }

        void FuncDecl::copy(FIRNode::Ptr node) {
            const auto funcDecl = to<FuncDecl>(node);
            FIRNode::copy(funcDecl);
            name = funcDecl->name->clone<Identifier>();
            for (const auto &genericParam : funcDecl->genericParams) {
                genericParams.push_back(genericParam->clone<GenericParam>());
            }
            for (const auto &arg : funcDecl->args) {
                args.push_back(arg->clone<Argument>());
            }
            for (const auto &result : funcDecl->results) {
                results.push_back(result->clone<IdentDecl>());
            }
            if (funcDecl->body) {
                body = funcDecl->body->clone<StmtBlock>();
            }
            type = funcDecl->type;
            originalName = funcDecl->originalName;
        }

        FIRNode::Ptr FuncDecl::cloneNode() {
            const auto node = std::make_shared<FuncDecl>();
            node->copy(shared_from_this());
            return node;
        }

        void VarDecl::copy(FIRNode::Ptr node) {
            const auto varDecl = to<VarDecl>(node);
            Stmt::copy(varDecl);
            name = varDecl->name->clone<Identifier>();
            if (varDecl->type) {
                type = varDecl->type->clone<Type>();
            }
            if (varDecl->initVal) {
                initVal = varDecl->initVal->clone<Expr>();
            }
        }

        FIRNode::Ptr VarDecl::cloneNode() {
            const auto node = std::make_shared<VarDecl>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr ConstDecl::cloneNode() {
            const auto node = std::make_shared<ConstDecl>();
            node->copy(shared_from_this());
            return node;
        }

        void WhileStmt::copy(FIRNode::Ptr node) {
            const auto whileStmt = to<WhileStmt>(node);
            Stmt::copy(whileStmt);
            cond = whileStmt->cond->clone<Expr>();
            body = whileStmt->body->clone<StmtBlock>();
        }

        FIRNode::Ptr WhileStmt::cloneNode() {
            const auto node = std::make_shared<WhileStmt>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr DoWhileStmt::cloneNode() {
            const auto node = std::make_shared<DoWhileStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void IfStmt::copy(FIRNode::Ptr node) {
            const auto ifStmt = to<IfStmt>(node);
            Stmt::copy(ifStmt);
            cond = ifStmt->cond->clone<Expr>();
            ifBody = ifStmt->ifBody->clone<Stmt>();
            if (ifStmt->elseBody) {
                elseBody = ifStmt->elseBody->clone<Stmt>();
            }
        }

        FIRNode::Ptr IfStmt::cloneNode() {
            const auto node = std::make_shared<IfStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void IndexSetDomain::copy(FIRNode::Ptr node) {
            const auto indexSetDomain = to<IndexSetDomain>(node);
            ForDomain::copy(indexSetDomain);
            set = indexSetDomain->set->clone<SetIndexSet>();
        }

        FIRNode::Ptr IndexSetDomain::cloneNode() {
            const auto node = std::make_shared<IndexSetDomain>();
            node->copy(shared_from_this());
            return node;
        }

        void RangeDomain::copy(FIRNode::Ptr node) {
            const auto rangeDomain = to<RangeDomain>(node);
            ForDomain::copy(rangeDomain);
            lower = rangeDomain->lower->clone<Expr>();
            upper = rangeDomain->upper->clone<Expr>();
        }

        FIRNode::Ptr RangeDomain::cloneNode() {
            const auto node = std::make_shared<RangeDomain>();
            node->copy(shared_from_this());
            return node;
        }

        void ForStmt::copy(FIRNode::Ptr node) {
            const auto forStmt = to<ForStmt>(node);
            Stmt::copy(forStmt);
            loopVar = forStmt->loopVar->clone<Identifier>();
            domain = forStmt->domain->clone<ForDomain>();
            body = forStmt->body->clone<StmtBlock>();
            stmt_label = forStmt->stmt_label;
        }

        FIRNode::Ptr ForStmt::cloneNode() {
            const auto node = std::make_shared<ForStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void NameNode::copy(FIRNode::Ptr node) {
            const auto name_node = to<NameNode>(node);
            Stmt::copy(name_node);
            body = name_node->body->clone<StmtBlock>();
        }

        FIRNode::Ptr NameNode::cloneNode() {
            const auto node = std::make_shared<NameNode>();
            node->copy(shared_from_this());
            return node;
        }

        void PrintStmt::copy(FIRNode::Ptr node) {
            const auto printStmt = to<PrintStmt>(node);
            Stmt::copy(printStmt);
            for (const auto &arg : printStmt->args) {
                args.push_back(arg->clone<Expr>());
            }
            printNewline = printStmt->printNewline;
        }

        FIRNode::Ptr PrintStmt::cloneNode() {
            const auto node = std::make_shared<PrintStmt>();
            node->copy(shared_from_this());
            return node;
        }


        void BreakStmt::copy(FIRNode::Ptr node) {
            const auto printStmt = to<BreakStmt>(node);
            Stmt::copy(printStmt);
        }

        FIRNode::Ptr BreakStmt::cloneNode() {
            const auto node = std::make_shared<BreakStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void ExprStmt::copy(FIRNode::Ptr node) {
            const auto exprStmt = to<ExprStmt>(node);
            Stmt::copy(exprStmt);
            expr = exprStmt->expr->clone<Expr>();
            stmt_label = exprStmt->stmt_label;
        }

        FIRNode::Ptr ExprStmt::cloneNode() {
            const auto node = std::make_shared<ExprStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void AssignStmt::copy(FIRNode::Ptr node) {
            const auto assignStmt = to<AssignStmt>(node);
            ExprStmt::copy(assignStmt);
            for (const auto &left : assignStmt->lhs) {
                lhs.push_back(left->clone<Expr>());
            }
            stmt_label = assignStmt->stmt_label;
        }

        FIRNode::Ptr AssignStmt::cloneNode() {
            const auto node = std::make_shared<AssignStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void ReduceStmt::copy(FIRNode::Ptr node) {
            const auto reduce_stmt = to<ReduceStmt>(node);
            ExprStmt::copy(reduce_stmt);
            for (const auto &left : reduce_stmt->lhs) {
                lhs.push_back(left->clone<Expr>());
            }
            stmt_label = reduce_stmt->stmt_label;
            reduction_op = reduce_stmt->reduction_op;
        }

        FIRNode::Ptr ReduceStmt::cloneNode() {
            const auto node = std::make_shared<ReduceStmt>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr Slice::cloneNode() {
            const auto node = std::make_shared<Slice>();
            node->copy(shared_from_this());
            return node;
        }

        void ExprParam::copy(FIRNode::Ptr node) {
            const auto exprParam = to<ExprParam>(node);
            ReadParam::copy(exprParam);
            expr = exprParam->expr->clone<Expr>();
        }

        FIRNode::Ptr ExprParam::cloneNode() {
            const auto node = std::make_shared<ExprParam>();
            node->copy(shared_from_this());
            return node;
        }

        void MapExpr::copy(FIRNode::Ptr node) {
            const auto mapExpr = to<MapExpr>(node);
            Expr::copy(mapExpr);
            func = mapExpr->func->clone<Identifier>();
            for (const auto &arg : mapExpr->partialActuals) {
                partialActuals.push_back(arg->clone<Expr>());
            }
            target = mapExpr->target->clone<SetIndexSet>();
        }

        void ReducedMapExpr::copy(FIRNode::Ptr node) {
            const auto reducedMapExpr = to<ReducedMapExpr>(node);
            MapExpr::copy(reducedMapExpr);
            op = reducedMapExpr->op;
        }

        FIRNode::Ptr ReducedMapExpr::cloneNode() {
            const auto node = std::make_shared<ReducedMapExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr UnreducedMapExpr::cloneNode() {
            const auto node = std::make_shared<UnreducedMapExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void UnaryExpr::copy(FIRNode::Ptr node) {
            const auto unaryExpr = to<UnaryExpr>(node);
            Expr::copy(unaryExpr);
            operand = unaryExpr->operand->clone<Expr>();
        }

        void BinaryExpr::copy(FIRNode::Ptr node) {
            const auto binaryExpr = to<BinaryExpr>(node);
            Expr::copy(binaryExpr);
            lhs = binaryExpr->lhs->clone<Expr>();
            rhs = binaryExpr->rhs->clone<Expr>();
        }

        void NaryExpr::copy(FIRNode::Ptr node) {
            const auto naryExpr = to<NaryExpr>(node);
            Expr::copy(naryExpr);
            for (const auto &operand : naryExpr->operands) {
                operands.push_back(operand->clone<Expr>());
            }
        }

        FIRNode::Ptr OrExpr::cloneNode() {
            const auto node = std::make_shared<OrExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr AndExpr::cloneNode() {
            const auto node = std::make_shared<AndExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr XorExpr::cloneNode() {
            const auto node = std::make_shared<XorExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void EqExpr::copy(FIRNode::Ptr node) {
            const auto eqExpr = to<EqExpr>(node);
            NaryExpr::copy(eqExpr);
            ops = eqExpr->ops;
        }

        FIRNode::Ptr EqExpr::cloneNode() {
            const auto node = std::make_shared<EqExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr NotExpr::cloneNode() {
            const auto node = std::make_shared<NotExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr AddExpr::cloneNode() {
            const auto node = std::make_shared<AddExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr SubExpr::cloneNode() {
            const auto node = std::make_shared<SubExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr MulExpr::cloneNode() {
            const auto node = std::make_shared<MulExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr DivExpr::cloneNode() {
            const auto node = std::make_shared<DivExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr ElwiseMulExpr::cloneNode() {
            const auto node = std::make_shared<ElwiseMulExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr ElwiseDivExpr::cloneNode() {
            const auto node = std::make_shared<ElwiseDivExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr LeftDivExpr::cloneNode() {
            const auto node = std::make_shared<LeftDivExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void NegExpr::copy(FIRNode::Ptr node) {
            const auto negExpr = to<NegExpr>(node);
            UnaryExpr::copy(negExpr);
            negate = negExpr->negate;
        }

        FIRNode::Ptr NegExpr::cloneNode() {
            const auto node = std::make_shared<NegExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr ExpExpr::cloneNode() {
            const auto node = std::make_shared<ExpExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr TransposeExpr::cloneNode() {
            const auto node = std::make_shared<TransposeExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void CallExpr::copy(FIRNode::Ptr node) {
            const auto callExpr = to<CallExpr>(node);
            Expr::copy(callExpr);
            func = callExpr->func->clone<Identifier>();
            for (const auto &genericArg : callExpr->genericArgs) {
                genericArgs.push_back(genericArg->clone<IndexSet>());
            }

            for (const auto &arg : callExpr->functorArgs) {
                functorArgs.push_back(arg ? arg->clone<Expr>() : Expr::Ptr());
            }


            for (const auto &arg : callExpr->args) {
                args.push_back(arg ? arg->clone<Expr>() : Expr::Ptr());
            }
        }

        FIRNode::Ptr CallExpr::cloneNode() {
            const auto node = std::make_shared<CallExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void TensorReadExpr::copy(FIRNode::Ptr node) {
            const auto tensorReadExpr = to<TensorReadExpr>(node);
            Expr::copy(tensorReadExpr);
            tensor = tensorReadExpr->tensor->clone<Expr>();
            for (const auto &index : tensorReadExpr->indices) {
                indices.push_back(index->clone<ReadParam>());
            }
        }

        FIRNode::Ptr TensorReadExpr::cloneNode() {
            const auto node = std::make_shared<TensorReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void SetReadExpr::copy(FIRNode::Ptr node) {
            const auto setReadExpr = to<SetReadExpr>(node);
            Expr::copy(setReadExpr);
            set = setReadExpr->set->clone<Expr>();
            for (const auto &index : setReadExpr->indices) {
                indices.push_back(index->clone<Expr>());
            }
        }

        FIRNode::Ptr SetReadExpr::cloneNode() {
            const auto node = std::make_shared<SetReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void TupleReadExpr::copy(FIRNode::Ptr node) {
            const auto tupleReadExpr = to<TupleReadExpr>(node);
            Expr::copy(tupleReadExpr);
            tuple = tupleReadExpr->tuple->clone<Expr>();
        }

        void NamedTupleReadExpr::copy(FIRNode::Ptr node) {
            const auto tupleReadExpr = to<NamedTupleReadExpr>(node);
            TupleReadExpr::copy(tupleReadExpr);
            elem = tupleReadExpr->elem->clone<Identifier>();
        }

        FIRNode::Ptr NamedTupleReadExpr::cloneNode() {
            const auto node = std::make_shared<NamedTupleReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void UnnamedTupleReadExpr::copy(FIRNode::Ptr node) {
            const auto tupleReadExpr = to<UnnamedTupleReadExpr>(node);
            TupleReadExpr::copy(tupleReadExpr);
            index = tupleReadExpr->index->clone<Expr>();
        }

        FIRNode::Ptr UnnamedTupleReadExpr::cloneNode() {
            const auto node = std::make_shared<UnnamedTupleReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void FieldReadExpr::copy(FIRNode::Ptr node) {
            const auto fieldReadExpr = to<FieldReadExpr>(node);
            Expr::copy(fieldReadExpr);
            setOrElem = fieldReadExpr->setOrElem->clone<Expr>();
            field = fieldReadExpr->field->clone<Identifier>();
        }

        FIRNode::Ptr FieldReadExpr::cloneNode() {
            const auto node = std::make_shared<FieldReadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void ParenExpr::copy(FIRNode::Ptr node) {
            const auto parenExpr = to<ParenExpr>(node);
            Expr::copy(parenExpr);
            expr = parenExpr->expr->clone<Expr>();
        }

        FIRNode::Ptr ParenExpr::cloneNode() {
            const auto node = std::make_shared<ParenExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void VarExpr::copy(FIRNode::Ptr node) {
            const auto varExpr = to<VarExpr>(node);
            Expr::copy(varExpr);
            ident = varExpr->ident;
        }

        FIRNode::Ptr VarExpr::cloneNode() {
            const auto node = std::make_shared<VarExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr RangeConst::cloneNode() {
            const auto node = std::make_shared<RangeConst>();
            node->copy(shared_from_this());
            return node;
        }

        void IntLiteral::copy(FIRNode::Ptr node) {
            const auto intLiteral = to<IntLiteral>(node);
            TensorLiteral::copy(intLiteral);
            val = intLiteral->val;
        }

        FIRNode::Ptr IntLiteral::cloneNode() {
            const auto node = std::make_shared<IntLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void FloatLiteral::copy(FIRNode::Ptr node) {
            const auto floatLiteral = to<FloatLiteral>(node);
            TensorLiteral::copy(floatLiteral);
            val = floatLiteral->val;
        }

        FIRNode::Ptr FloatLiteral::cloneNode() {
            const auto node = std::make_shared<FloatLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void BoolLiteral::copy(FIRNode::Ptr node) {
            const auto boolLiteral = to<BoolLiteral>(node);
            TensorLiteral::copy(boolLiteral);
            val = boolLiteral->val;
        }

        FIRNode::Ptr BoolLiteral::cloneNode() {
            const auto node = std::make_shared<BoolLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void StringLiteral::copy(FIRNode::Ptr node) {
            const auto stringLiteral = to<StringLiteral>(node);
            TensorLiteral::copy(stringLiteral);
            val = stringLiteral->val;
        }

        FIRNode::Ptr StringLiteral::cloneNode() {
            const auto node = std::make_shared<StringLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void DenseTensorLiteral::copy(FIRNode::Ptr node) {
            const auto denseTensorLiteral = to<DenseTensorLiteral>(node);
            TensorLiteral::copy(denseTensorLiteral);
            transposed = denseTensorLiteral->transposed;
        }

        void IntVectorLiteral::copy(FIRNode::Ptr node) {
            const auto intVectorLiteral = to<IntVectorLiteral>(node);
            DenseTensorLiteral::copy(intVectorLiteral);
            vals = intVectorLiteral->vals;
        }

        FIRNode::Ptr IntVectorLiteral::cloneNode() {
            const auto node = std::make_shared<IntVectorLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void FloatVectorLiteral::copy(FIRNode::Ptr node) {
            const auto floatVectorLiteral = to<FloatVectorLiteral>(node);
            DenseTensorLiteral::copy(floatVectorLiteral);
            vals = floatVectorLiteral->vals;
        }

        FIRNode::Ptr FloatVectorLiteral::cloneNode() {
            const auto node = std::make_shared<FloatVectorLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void NDTensorLiteral::copy(FIRNode::Ptr node) {
            const auto ndTensorLiteral = to<NDTensorLiteral>(node);
            DenseTensorLiteral::copy(ndTensorLiteral);
            for (const auto &elem : ndTensorLiteral->elems) {
                elems.push_back(elem->clone<DenseTensorLiteral>());
            }
        }

        FIRNode::Ptr NDTensorLiteral::cloneNode() {
            const auto node = std::make_shared<NDTensorLiteral>();
            node->copy(shared_from_this());
            return node;
        }

        void ApplyStmt::copy(FIRNode::Ptr node) {
            const auto applyStmt = to<ApplyStmt>(node);
            Stmt::copy(applyStmt);
            map = applyStmt->map->clone<UnreducedMapExpr>();
        }

        FIRNode::Ptr ApplyStmt::cloneNode() {
            const auto node = std::make_shared<ApplyStmt>();
            node->copy(shared_from_this());
            return node;
        }

        void Test::copy(FIRNode::Ptr node) {
            const auto test = to<Test>(node);
            FIRNode::copy(test);
            func = test->func->clone<Identifier>();
            for (const auto &arg : test->args) {
                args.push_back(arg->clone<Expr>());
            }
            expected = test->expected->clone<Expr>();
        }

        FIRNode::Ptr Test::cloneNode() {
            const auto node = std::make_shared<Test>();
            node->copy(shared_from_this());
            return node;
        }

        //GraphIt additions

        void VertexSetType::copy(FIRNode::Ptr node) {
            const auto vertexSetType = to<VertexSetType>(node);
            Type::copy(vertexSetType);
            element = vertexSetType->element->clone<ElementType>();
        }
        FIRNode::Ptr VertexSetType::cloneNode() {
            const auto node = std::make_shared<VertexSetType>();
            node->copy(shared_from_this());
            return node;
        }


        void ListType::copy(FIRNode::Ptr node) {
            const auto listType = to<ListType>(node);
            Type::copy(listType);
            list_element_type = listType->list_element_type->clone<Type>();
        }
        FIRNode::Ptr ListType::cloneNode() {
            const auto node = std::make_shared<ListType>();
            node->copy(shared_from_this());
            return node;
        }

        void EdgeSetType::copy(FIRNode::Ptr node) {
            const auto edgeSetType = to<EdgeSetType>(node);
            Type::copy(edgeSetType);
            edge_element_type = edgeSetType->edge_element_type->clone<ElementType>();
        }
        FIRNode::Ptr EdgeSetType::cloneNode() {
            // TODO: need to add some support for the list of vertex element list
            const auto node = std::make_shared<EdgeSetType>();
            node->copy(shared_from_this());
            return node;
        }


        void VertexSetAllocExpr::copy(FIRNode::Ptr node) {
            //TODO: figure out what the copy operator should do
            const auto vertexset_set_alloc_expr = to<VertexSetAllocExpr>(node);
            Expr::copy(vertexset_set_alloc_expr);
            elementType = vertexset_set_alloc_expr->elementType->clone<ElementType>();
            numElements = vertexset_set_alloc_expr->numElements->clone<Expr>();
        }

        FIRNode::Ptr VertexSetAllocExpr::cloneNode() {
            const auto node = std::make_shared<VertexSetAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void IntersectionExpr::copy(FIRNode::Ptr node) {
            const auto intersection_expr = to<IntersectionExpr>(node);
            Expr::copy(intersection_expr);
            vertex_a = intersection_expr->vertex_a->clone<Expr>();
            vertex_b = intersection_expr->vertex_b->clone<Expr>();
            numA = intersection_expr->numA->clone<Expr>();
            numB = intersection_expr->numB->clone<Expr>();
            reference = intersection_expr->reference->clone<Expr>();
        }


        FIRNode::Ptr IntersectionExpr::cloneNode() {
            const auto node = std::make_shared<IntersectionExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void IntersectNeighborExpr::copy(FIRNode::Ptr node) {
            const auto intersection_expr = to<IntersectNeighborExpr>(node);
            Expr::copy(intersection_expr);
            edges = intersection_expr->edges->clone<Expr>();
            vertex_a = intersection_expr->vertex_a->clone<Expr>();
            vertex_b = intersection_expr->vertex_b->clone<Expr>();
        }


        FIRNode::Ptr IntersectNeighborExpr::cloneNode() {
            const auto node = std::make_shared<IntersectNeighborExpr>();
            node->copy(shared_from_this());
            return node;
        }


        void EdgeSetLoadExpr::copy(FIRNode::Ptr node) {
            //TODO: figure out what the copy operator should do
            const auto edge_set_load_expr = to<EdgeSetLoadExpr>(node);
            Expr::copy(edge_set_load_expr);
            //element_type = edge_set_load_expr->element_type->clone<ElementType>();
            file_name = edge_set_load_expr->file_name->clone<Expr>();
        }


        FIRNode::Ptr EdgeSetLoadExpr::cloneNode() {
            const auto node = std::make_shared<EdgeSetLoadExpr>();
            node->copy(shared_from_this());
            return node;
        }

        FIRNode::Ptr ListAllocExpr::cloneNode() {
            const auto node = std::make_shared<ListAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void ListAllocExpr::copy(FIRNode::Ptr node) {
            const auto list_alloc_expr = to<ListAllocExpr>(node);
            Expr::copy(list_alloc_expr);
            general_element_type = list_alloc_expr->general_element_type->clone<Type>();
        }


        FIRNode::Ptr VectorAllocExpr::cloneNode() {
            const auto node = std::make_shared<VectorAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void VectorAllocExpr::copy(FIRNode::Ptr node) {
            const auto vector_alloc_expr = to<VectorAllocExpr>(node);
            Expr::copy(vector_alloc_expr);
            general_element_type = vector_alloc_expr->general_element_type->clone<Type>();
            elementType = vector_alloc_expr->elementType->clone<ElementType>();
            vector_scalar_type = vector_alloc_expr->vector_scalar_type->clone<ScalarType>();
        }

        void MethodCallExpr::copy(FIRNode::Ptr node) {
            const auto method_call_expr = to<MethodCallExpr>(node);
            Expr::copy(method_call_expr);
            method_name = method_call_expr->method_name->clone<Identifier>();
            target = method_call_expr->target->clone<Expr>();

//            for (const auto &genericArg : method_call_expr->genericArgs) {
//                genericArgs.push_back(genericArg->clone<IndexSet>());
//            }
            for (const auto &arg : method_call_expr->args) {
                args.push_back(arg ? arg->clone<Expr>() : Expr::Ptr());
            }
        }


        FIRNode::Ptr MethodCallExpr::cloneNode() {
            const auto node = std::make_shared<MethodCallExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void FuncExpr::copy(FIRNode::Ptr node) {
            const auto funcExpr = to<FuncExpr>(node);
            Expr::copy(funcExpr);
            name = funcExpr->name->clone<Identifier>();
            for (const auto &arg : funcExpr->args) {
                args.push_back(arg ? arg->clone<Expr>() : Expr::Ptr());
            }

        }


        FIRNode::Ptr FuncExpr::cloneNode() {
            const auto node = std::make_shared<FuncExpr>();
            node->copy(shared_from_this());
            return node;
        }



        void ApplyExpr::copy(FIRNode::Ptr node) {
            const auto apply_expr = to<ApplyExpr>(node);
            Expr::copy(apply_expr);
            target = apply_expr->target->clone<Expr>();
            input_function = apply_expr->input_function->clone<FuncExpr>();
            type = apply_expr->type;

            if (apply_expr->from_expr){
                from_expr = apply_expr->from_expr->clone<FromExpr>();
            }
            if (apply_expr->to_expr){
                to_expr = apply_expr->to_expr->clone<ToExpr>();
            }
        }

        FIRNode::Ptr ApplyExpr::cloneNode() {
            const auto node = std::make_shared<ApplyExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void WhereExpr::copy(FIRNode::Ptr node) {
            const auto where_expr = to<WhereExpr>(node);
            Expr::copy(where_expr);
            target = where_expr->target->clone<Expr>();
            input_func = where_expr->input_func->clone<FuncExpr>();
        }

        FIRNode::Ptr WhereExpr::cloneNode() {
            const auto node = std::make_shared<WhereExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void FromExpr::copy(FIRNode::Ptr node) {
            const auto from_expr = to<FromExpr>(node);
            Expr::copy(from_expr);
            input_func = from_expr->input_func->clone<FuncExpr>();
       }

        FIRNode::Ptr FromExpr::cloneNode() {
            const auto node = std::make_shared<FromExpr>();
            node->copy(shared_from_this());
            return node;
        }

        void ToExpr::copy(FIRNode::Ptr node) {
            const auto from_expr = to<ToExpr>(node);
            Expr::copy(from_expr);
            input_func = from_expr->input_func->clone<FuncExpr>();
        }

        FIRNode::Ptr ToExpr::cloneNode() {
            const auto node = std::make_shared<ToExpr>();
            node->copy(shared_from_this());
            return node;
        }


        TensorType::Ptr makeTensorType(ScalarType::Type componentType,
                                       const TensorDimensions &dimensions,
                                       bool transposed) {
            const auto scalarType = std::make_shared<ScalarType>();
            scalarType->type = componentType;

            if (dimensions.empty()) {
                return scalarType;
            }

            TensorType::Ptr retType = scalarType;
            for (unsigned i = 0; i < dimensions[0].size(); ++i) {
                const auto tensorType = std::make_shared<NDTensorType>();
                tensorType->blockType = retType;

                const unsigned idx = dimensions[0].size() - i - 1;
                for (unsigned j = 0; j < dimensions.size(); ++j) {
                    tensorType->indexSets.push_back(dimensions[j][idx]);
                }

                retType = tensorType;
            }
            to<NDTensorType>(retType)->transposed = transposed;

            return retType;
        }

	// OG Additions

        void PriorityQueueType::copy(FIRNode::Ptr node) {
            const auto priorityQueueType = to<PriorityQueueType>(node);
            Type::copy(priorityQueueType);
            element = priorityQueueType->element->clone<ElementType>();
            priority_type = priorityQueueType->element->clone<ScalarType>();

        }
        FIRNode::Ptr PriorityQueueType::cloneNode() {
            const auto node = std::make_shared<PriorityQueueType>();
            node->copy(shared_from_this());
            return node;
        }

        void PriorityQueueAllocExpr::copy(FIRNode::Ptr node) {
            //TODO: figure out what the copy operator should do
            const auto priority_queue_alloc_expr = to<PriorityQueueAllocExpr>(node);
            Expr::copy(priority_queue_alloc_expr);
            elementType = priority_queue_alloc_expr->elementType->clone<ElementType>();
            numElements = priority_queue_alloc_expr->numElements->clone<Expr>();
	    
            dup_within_bucket = priority_queue_alloc_expr->dup_within_bucket->clone<Expr>();
            dup_across_bucket = priority_queue_alloc_expr->dup_across_bucket->clone<Expr>();
            vector_function = priority_queue_alloc_expr->vector_function->clone<Identifier>(); 
            bucket_ordering = priority_queue_alloc_expr->bucket_ordering->clone<Expr>();
            priority_ordering = priority_queue_alloc_expr->priority_ordering->clone<Expr>();
            init_bucket = priority_queue_alloc_expr->init_bucket->clone<Expr>();
            starting_node = priority_queue_alloc_expr->starting_node->clone<Expr>();


        }

        FIRNode::Ptr PriorityQueueAllocExpr::cloneNode() {
            const auto node = std::make_shared<PriorityQueueAllocExpr>();
            node->copy(shared_from_this());
            return node;
        }


    }
}
