//
// Created by Yunming Zhang on 1/24/17.
//


#ifndef GRAPHIT_FIR_H
#define GRAPHIT_FIR_H


#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>

#include <graphit/frontend/scanner.h>
#include <graphit/frontend/fir_visitor.h>


namespace graphit {
    namespace fir {

        struct FIRNode;
        struct SetType;

        template<typename T>
        inline bool isa(std::shared_ptr<FIRNode> ptr) {
            return (bool) std::dynamic_pointer_cast<T>(ptr);
        }

        template<typename T>
        inline const std::shared_ptr<T> to(std::shared_ptr<FIRNode> ptr) {
            std::shared_ptr<T> ret = std::dynamic_pointer_cast<T>(ptr);
            assert(ret != nullptr);
            return ret;
        }

// Base class for front-end intermediate representation.
        struct FIRNode : public std::enable_shared_from_this<FIRNode> {
            typedef std::shared_ptr<FIRNode> Ptr;

            FIRNode() : lineBegin(0), colBegin(0), lineEnd(0), colEnd(0) {}

            template<typename T = FIRNode>
            std::shared_ptr<T> clone() {
                return to<T>(cloneNode());
            }

            virtual void accept(FIRVisitor *) = 0;

            virtual unsigned getLineBegin() { return lineBegin; }

            virtual unsigned getColBegin() { return colBegin; }

            virtual unsigned getLineEnd() { return lineEnd; }

            virtual unsigned getColEnd() { return colEnd; }

            void setBeginLoc(const Token &);

            void setEndLoc(const Token &);

            void setLoc(const Token &);

            friend std::ostream &operator<<(std::ostream &, FIRNode &);

        protected:
            template<typename T = FIRNode>
            std::shared_ptr<T> self() {
                return to<T>(shared_from_this());
            }

            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode() = 0;

        private:
            unsigned lineBegin;
            unsigned colBegin;
            unsigned lineEnd;
            unsigned colEnd;
        };

        struct Program : public FIRNode {
            std::vector<FIRNode::Ptr> elems;

            typedef std::shared_ptr<Program> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Program>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct Stmt : public FIRNode {
            typedef std::shared_ptr<Stmt> Ptr;
            std::string stmt_label = "";
        };

        struct StmtBlock : public Stmt {
            std::vector<Stmt::Ptr> stmts;

            typedef std::shared_ptr<StmtBlock> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<StmtBlock>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct Type : public FIRNode {
            typedef std::shared_ptr<Type> Ptr;
        };

        struct Expr : public FIRNode {
            typedef std::shared_ptr<Expr> Ptr;
        };

        struct IndexSet : public FIRNode {
            typedef std::shared_ptr<IndexSet> Ptr;
        };

        struct RangeIndexSet : public IndexSet {
            int range = 0;

            typedef std::shared_ptr<RangeIndexSet> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<RangeIndexSet>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct SetIndexSet : public IndexSet {
            std::string setName;
            std::shared_ptr<SetType> setDef; // Reference to original definition of set.

            typedef std::shared_ptr<SetIndexSet> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<SetIndexSet>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct GenericIndexSet : public SetIndexSet {
            enum class Type {
                UNKNOWN, RANGE
            };

            Type type;

            typedef std::shared_ptr<GenericIndexSet> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<GenericIndexSet>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct DynamicIndexSet : public IndexSet {
            typedef std::shared_ptr<DynamicIndexSet> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<DynamicIndexSet>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct ElementType : public Type {
            std::string ident;
            SetIndexSet::Ptr source; // Reference to inferred source index set.

            typedef std::shared_ptr<ElementType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ElementType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct Endpoint : public FIRNode {
            SetIndexSet::Ptr set;
            ElementType::Ptr element;

            typedef std::shared_ptr<Endpoint> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Endpoint>());
            }

            virtual unsigned getLineBegin() { return set->getLineBegin(); }

            virtual unsigned getColBegin() { return set->getColBegin(); }

            virtual unsigned getLineEnd() { return set->getLineEnd(); }

            virtual unsigned getColEnd() { return set->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct SetType : public Type {
            ElementType::Ptr element;

            typedef std::shared_ptr<SetType> Ptr;

            static Ptr getUndefinedSetType();

        protected:
            virtual void copy(FIRNode::Ptr);
        };

        struct UnstructuredSetType : public SetType {
            typedef std::shared_ptr<UnstructuredSetType> Ptr;

            virtual bool isHomogeneous() const { return true; }

            virtual size_t getArity() const { return 0; }

            virtual Endpoint::Ptr getEndpoint(size_t) const { return Endpoint::Ptr(); }
        };

        struct TupleLength : public FIRNode {
            int val = 0;

            typedef std::shared_ptr<TupleLength> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<TupleLength>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct HomogeneousEdgeSetType : public UnstructuredSetType {
            Endpoint::Ptr endpoint;
            TupleLength::Ptr arity;

            typedef std::shared_ptr<HomogeneousEdgeSetType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<HomogeneousEdgeSetType>());
            }

            virtual size_t getArity() const { return arity->val; }

            virtual Endpoint::Ptr getEndpoint(size_t i) const { return endpoint; }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct HeterogeneousEdgeSetType : public UnstructuredSetType {
            std::vector<Endpoint::Ptr> endpoints;

            typedef std::shared_ptr<HeterogeneousEdgeSetType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<HeterogeneousEdgeSetType>());
            }

            virtual bool isHomogeneous() const;

            virtual size_t getArity() const { return endpoints.size(); }

            virtual Endpoint::Ptr getEndpoint(size_t i) const { return endpoints.at(i); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct GridSetType : public SetType {
            Endpoint::Ptr underlyingPointSet;
            size_t dimensions;

            typedef std::shared_ptr<GridSetType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<GridSetType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct TupleType : public Type {
            typedef std::shared_ptr<TupleType> Ptr;
        };

        struct Identifier : public FIRNode {
            std::string ident;

            typedef std::shared_ptr<Identifier> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Identifier>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct TupleElement : public FIRNode {
            Identifier::Ptr name;
            ElementType::Ptr element;

            typedef std::shared_ptr<TupleElement> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<TupleElement>());
            }

            virtual unsigned getLineBegin() { return name->getLineBegin(); }

            virtual unsigned getColBegin() { return name->getColBegin(); }

            virtual unsigned getLineEnd() { return element->getLineEnd(); }

            virtual unsigned getColEnd() { return element->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct NamedTupleType : public TupleType {
            std::vector<TupleElement::Ptr> elems;

            typedef std::shared_ptr<NamedTupleType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<NamedTupleType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct UnnamedTupleType : public TupleType {
            ElementType::Ptr element;
            TupleLength::Ptr length;

            typedef std::shared_ptr<UnnamedTupleType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<UnnamedTupleType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct TensorType : public Type {
            typedef std::shared_ptr<TensorType> Ptr;
        };

        struct ScalarType : public TensorType {
            enum class Type {
                INT, UINT, UINT_64, FLOAT, BOOL, DOUBLE, COMPLEX, STRING
            };

            Type type;

            typedef std::shared_ptr<ScalarType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ScalarType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct NDTensorType : public TensorType {
            std::vector<IndexSet::Ptr> indexSets;
            TensorType::Ptr blockType;
            bool transposed = false;
            //adding support for edge_element_type type in vectors
            ElementType::Ptr element;

            typedef std::shared_ptr<NDTensorType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<NDTensorType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        // A type for list. The element in list can be of any type, VertexsetType, TensorType ...
        struct ListType : public Type {

            Type::Ptr list_element_type;


            typedef std::shared_ptr<ListType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ListType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct OpaqueType : public Type {
            typedef std::shared_ptr<OpaqueType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<OpaqueType>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct IdentDecl : public FIRNode {
            Identifier::Ptr name;
            Type::Ptr type;

            typedef std::shared_ptr<IdentDecl> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<IdentDecl>());
            }

            virtual unsigned getLineBegin() { return name->getLineBegin(); }

            virtual unsigned getColBegin() { return name->getColBegin(); }

            virtual unsigned getLineEnd() { return type->getLineEnd(); }

            virtual unsigned getColEnd() { return type->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct FieldDecl : public IdentDecl {
            typedef std::shared_ptr<FieldDecl> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<FieldDecl>());
            }

            virtual unsigned getLineEnd() { return FIRNode::getLineEnd(); }

            virtual unsigned getColEnd() { return FIRNode::getColEnd(); }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct ElementTypeDecl : public FIRNode {
            Identifier::Ptr name;
            std::vector<FieldDecl::Ptr> fields;

            typedef std::shared_ptr<ElementTypeDecl> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ElementTypeDecl>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct Argument : public IdentDecl {
            typedef std::shared_ptr<Argument> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Argument>());
            }

            virtual bool isInOut() { return false; }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct InOutArgument : public Argument {
            typedef std::shared_ptr<InOutArgument> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<InOutArgument>());
            }

            virtual unsigned getLineBegin() { return FIRNode::getLineBegin(); }

            virtual unsigned getColBegin() { return FIRNode::getColBegin(); }

            virtual bool isInOut() { return true; }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct ExternDecl : public IdentDecl {
            typedef std::shared_ptr<ExternDecl> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ExternDecl>());
            }

            virtual unsigned getLineBegin() { return FIRNode::getLineBegin(); }

            virtual unsigned getColBegin() { return FIRNode::getColBegin(); }

            virtual unsigned getLineEnd() { return FIRNode::getLineEnd(); }

            virtual unsigned getColEnd() { return FIRNode::getColEnd(); }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct GenericParam : public FIRNode {
            enum class Type {
                UNKNOWN, RANGE
            };

            std::string name;
            Type type;

            typedef std::shared_ptr<GenericParam> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<GenericParam>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct FuncDecl : public FIRNode {
            enum class Type {
                INTERNAL, EXPORTED, EXTERNAL
            };

            Identifier::Ptr name;
            std::vector<GenericParam::Ptr> genericParams;
            std::vector<Argument::Ptr> functorArgs;
            std::vector<Argument::Ptr> args;
            std::vector<IdentDecl::Ptr> results;
            StmtBlock::Ptr body;
            Type type;
            std::string originalName;

            typedef std::shared_ptr<FuncDecl> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<FuncDecl>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct VarDecl : public Stmt {
            Identifier::Ptr name;
            Type::Ptr type;
            Expr::Ptr initVal;

            typedef std::shared_ptr<VarDecl> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<VarDecl>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ConstDecl : public VarDecl {
            typedef std::shared_ptr<ConstDecl> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ConstDecl>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct WhileStmt : public Stmt {
            Expr::Ptr cond;
            StmtBlock::Ptr body;

            typedef std::shared_ptr<WhileStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<WhileStmt>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct DoWhileStmt : public WhileStmt {
            typedef std::shared_ptr<DoWhileStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<DoWhileStmt>());
            }

            virtual unsigned getLineEnd() { return cond->getLineEnd(); }

            virtual unsigned getColEnd() { return cond->getColEnd(); }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct IfStmt : public Stmt {
            Expr::Ptr cond;
            Stmt::Ptr ifBody;
            Stmt::Ptr elseBody;

            typedef std::shared_ptr<IfStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<IfStmt>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ForDomain : public FIRNode {
            typedef std::shared_ptr<ForDomain> Ptr;
        };

        struct IndexSetDomain : public ForDomain {
            SetIndexSet::Ptr set;

            typedef std::shared_ptr<IndexSetDomain> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<IndexSetDomain>());
            }

            virtual unsigned getLineBegin() { return set->getLineBegin(); }

            virtual unsigned getColBegin() { return set->getColBegin(); }

            virtual unsigned getLineEnd() { return set->getLineEnd(); }

            virtual unsigned getColEnd() { return set->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct RangeDomain : public ForDomain {
            Expr::Ptr lower;
            Expr::Ptr upper;

            typedef std::shared_ptr<RangeDomain> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<RangeDomain>());
            }

            virtual unsigned getLineBegin() { return lower->getLineBegin(); }

            virtual unsigned getColBegin() { return lower->getColBegin(); }

            virtual unsigned getLineEnd() { return upper->getLineEnd(); }

            virtual unsigned getColEnd() { return upper->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ForStmt : public Stmt {
            Identifier::Ptr loopVar;
            ForDomain::Ptr domain;
            StmtBlock::Ptr body;

            typedef std::shared_ptr<ForStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ForStmt>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ParForStmt : public Stmt {
            Identifier::Ptr loopVar;
            ForDomain::Ptr domain;
            StmtBlock::Ptr body;

            typedef std::shared_ptr<ParForStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ParForStmt>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct NameNode : public Stmt {
            StmtBlock::Ptr body;

            typedef std::shared_ptr<NameNode> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<NameNode>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct PrintStmt : public Stmt {
            std::vector<Expr::Ptr> args;
            bool printNewline = false;

            typedef std::shared_ptr<PrintStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<PrintStmt>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct BreakStmt : public Stmt {

            typedef std::shared_ptr<BreakStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<BreakStmt>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ExprStmt : public Stmt {
            Expr::Ptr expr;

            typedef std::shared_ptr<ExprStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ExprStmt>());
            }

            virtual unsigned getLineBegin() { return expr->getLineBegin(); }

            virtual unsigned getColBegin() { return expr->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct AssignStmt : public ExprStmt {
            std::vector<Expr::Ptr> lhs;

            typedef std::shared_ptr<AssignStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<AssignStmt>());
            }

            virtual unsigned getLineBegin() { return lhs.front()->getLineBegin(); }

            virtual unsigned getColBegin() { return lhs.front()->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct ReduceStmt : public ExprStmt {
            std::vector<Expr::Ptr> lhs;
            enum class ReductionOp {
                MIN, SUM, MAX, ASYNC_MAX, ASYNC_MIN
            };
            ReductionOp reduction_op;

            typedef std::shared_ptr<ReduceStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ReduceStmt>());
            }

            virtual unsigned getLineBegin() { return lhs.front()->getLineBegin(); }

            virtual unsigned getColBegin() { return lhs.front()->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct ReadParam : public FIRNode {
            typedef std::shared_ptr<ReadParam> Ptr;

            virtual bool isSlice() { return false; }
        };

        struct Slice : public ReadParam {
            typedef std::shared_ptr<Slice> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Slice>());
            }

            virtual bool isSlice() { return true; }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct ExprParam : public ReadParam {
            Expr::Ptr expr;

            typedef std::shared_ptr<ExprParam> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ExprParam>());
            }

            virtual unsigned getLineBegin() { return expr->getLineBegin(); }

            virtual unsigned getColBegin() { return expr->getColBegin(); }

            virtual unsigned getLineEnd() { return expr->getLineEnd(); }

            virtual unsigned getColEnd() { return expr->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct MapExpr : public Expr {
            enum class ReductionOp {
                NONE, SUM
            };

            Identifier::Ptr func;
            std::vector<IndexSet::Ptr> genericArgs;
            std::vector<Expr::Ptr> partialActuals;
            SetIndexSet::Ptr target;
            SetIndexSet::Ptr through;

            typedef std::shared_ptr<MapExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<MapExpr>());
            }

            virtual ReductionOp getReductionOp() const = 0;

        protected:
            virtual void copy(FIRNode::Ptr);
        };

        struct ReducedMapExpr : public MapExpr {
            ReductionOp op;

            typedef std::shared_ptr<ReducedMapExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ReducedMapExpr>());
            }

            virtual ReductionOp getReductionOp() const { return op; }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct UnreducedMapExpr : public MapExpr {
            typedef std::shared_ptr<UnreducedMapExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<UnreducedMapExpr>());
            }

            virtual unsigned getLineEnd() { return target->getLineEnd(); }

            virtual unsigned getColEnd() { return target->getColEnd(); }

            virtual ReductionOp getReductionOp() const { return ReductionOp::NONE; }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };


        struct UnaryExpr : public Expr {
            Expr::Ptr operand;

            typedef std::shared_ptr<UnaryExpr> Ptr;

        protected:
            virtual void copy(FIRNode::Ptr);
        };

        struct BinaryExpr : public Expr {
            Expr::Ptr lhs;
            Expr::Ptr rhs;

            typedef std::shared_ptr<BinaryExpr> Ptr;

            virtual unsigned getLineBegin() { return lhs->getLineBegin(); }

            virtual unsigned getColBegin() { return lhs->getColBegin(); }

            virtual unsigned getLineEnd() { return rhs->getLineEnd(); }

            virtual unsigned getColEnd() { return rhs->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);
        };

        struct NaryExpr : public Expr {
            std::vector<Expr::Ptr> operands;

            typedef std::shared_ptr<NaryExpr> Ptr;

        protected:
            virtual void copy(FIRNode::Ptr);
        };

        struct OrExpr : public BinaryExpr {
            typedef std::shared_ptr<OrExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<OrExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct AndExpr : public BinaryExpr {
            typedef std::shared_ptr<AndExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<AndExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct XorExpr : public BinaryExpr {
            typedef std::shared_ptr<XorExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<XorExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct EqExpr : public NaryExpr {
            enum class Op {
                LT, LE, GT, GE, EQ, NE
            };

            std::vector<Op> ops;

            typedef std::shared_ptr<EqExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<EqExpr>());
            }

            virtual unsigned getLineBegin() { return operands.front()->getLineBegin(); }

            virtual unsigned getColBegin() { return operands.front()->getColBegin(); }

            virtual unsigned getLineEnd() { return operands.back()->getLineEnd(); }

            virtual unsigned getColEnd() { return operands.back()->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct NotExpr : public UnaryExpr {
            typedef std::shared_ptr<NotExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<NotExpr>());
            }

            virtual unsigned getLineEnd() { return operand->getLineEnd(); }

            virtual unsigned getColEnd() { return operand->getColEnd(); }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct AddExpr : public BinaryExpr {
            typedef std::shared_ptr<AddExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<AddExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct SubExpr : public BinaryExpr {
            typedef std::shared_ptr<SubExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<SubExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct MulExpr : public BinaryExpr {
            typedef std::shared_ptr<MulExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<MulExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct DivExpr : public BinaryExpr {
            typedef std::shared_ptr<DivExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<DivExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct ElwiseMulExpr : public BinaryExpr {
            typedef std::shared_ptr<ElwiseMulExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ElwiseMulExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct ElwiseDivExpr : public BinaryExpr {
            typedef std::shared_ptr<ElwiseDivExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ElwiseDivExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct LeftDivExpr : public BinaryExpr {
            typedef std::shared_ptr<LeftDivExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<LeftDivExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct NegExpr : public UnaryExpr {
            bool negate = false;

            typedef std::shared_ptr<NegExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<NegExpr>());
            }

            virtual unsigned getLineEnd() { return operand->getLineEnd(); }

            virtual unsigned getColEnd() { return operand->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ExpExpr : public BinaryExpr {
            typedef std::shared_ptr<ExpExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ExpExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct TransposeExpr : public UnaryExpr {
            typedef std::shared_ptr<TransposeExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<TransposeExpr>());
            }

            virtual unsigned getLineBegin() { return operand->getLineBegin(); }

            virtual unsigned getColBegin() { return operand->getColBegin(); }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct CallExpr : public Expr {
            Identifier::Ptr func;
            std::vector<IndexSet::Ptr> genericArgs;
            std::vector<Expr::Ptr> args;
            std::vector<Expr::Ptr> functorArgs;

            typedef std::shared_ptr<CallExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<CallExpr>());
            }

            virtual unsigned getLineBegin() { return func->getLineBegin(); }

            virtual unsigned getColBegin() { return func->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct TensorReadExpr : public Expr {
            Expr::Ptr tensor;
            std::vector<ReadParam::Ptr> indices;

            typedef std::shared_ptr<TensorReadExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<TensorReadExpr>());
            }

            virtual unsigned getLineBegin() { return tensor->getLineBegin(); }

            virtual unsigned getColBegin() { return tensor->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct SetReadExpr : public Expr {
            Expr::Ptr set;
            std::vector<Expr::Ptr> indices;

            typedef std::shared_ptr<SetReadExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<SetReadExpr>());
            }

            virtual unsigned getLineBegin() { return set->getLineBegin(); }

            virtual unsigned getColBegin() { return set->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct TupleReadExpr : public Expr {
            Expr::Ptr tuple;

            typedef std::shared_ptr<TupleReadExpr> Ptr;

            virtual unsigned getLineBegin() { return tuple->getLineBegin(); }

            virtual unsigned getColBegin() { return tuple->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);
        };

        struct NamedTupleReadExpr : public TupleReadExpr {
            Identifier::Ptr elem;

            typedef std::shared_ptr<NamedTupleReadExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<NamedTupleReadExpr>());
            }

            virtual unsigned getLineEnd() { return elem->getLineEnd(); }

            virtual unsigned getColEnd() { return elem->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct UnnamedTupleReadExpr : public TupleReadExpr {
            Expr::Ptr index;

            typedef std::shared_ptr<UnnamedTupleReadExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<UnnamedTupleReadExpr>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct FieldReadExpr : public Expr {
            Expr::Ptr setOrElem;
            Identifier::Ptr field;

            typedef std::shared_ptr<FieldReadExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<FieldReadExpr>());
            }

            virtual unsigned getLineBegin() { return setOrElem->getLineBegin(); }

            virtual unsigned getColBegin() { return setOrElem->getColBegin(); }

            virtual unsigned getLineEnd() { return field->getLineEnd(); }

            virtual unsigned getColEnd() { return field->getColEnd(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ParenExpr : public Expr {
            Expr::Ptr expr;

            typedef std::shared_ptr<ParenExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ParenExpr>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct VarExpr : public Expr {
            std::string ident;

            typedef std::shared_ptr<VarExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<VarExpr>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct RangeConst : public VarExpr {
            typedef std::shared_ptr<RangeConst> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<RangeConst>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
        };

        struct TensorLiteral : public Expr {
            typedef std::shared_ptr<TensorLiteral> Ptr;
        };

        struct IntLiteral : public TensorLiteral {
            int val = 0;

            typedef std::shared_ptr<IntLiteral> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<IntLiteral>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct FloatLiteral : public TensorLiteral {
            double val = 0.0;

            typedef std::shared_ptr<FloatLiteral> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<FloatLiteral>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct BoolLiteral : public TensorLiteral {
            bool val = false;

            typedef std::shared_ptr<BoolLiteral> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<BoolLiteral>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct StringLiteral : public TensorLiteral {
            std::string val;

            typedef std::shared_ptr<StringLiteral> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<StringLiteral>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct DenseTensorLiteral : public TensorLiteral {
            bool transposed = false;

            typedef std::shared_ptr<DenseTensorLiteral> Ptr;

        protected:
            virtual void copy(FIRNode::Ptr);
        };

        struct IntVectorLiteral : public DenseTensorLiteral {
            std::vector<int> vals;

            typedef std::shared_ptr<IntVectorLiteral> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<IntVectorLiteral>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct FloatVectorLiteral : public DenseTensorLiteral {
            std::vector<double> vals;

            typedef std::shared_ptr<FloatVectorLiteral> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<FloatVectorLiteral>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct NDTensorLiteral : public DenseTensorLiteral {
            std::vector<DenseTensorLiteral::Ptr> elems;

            typedef std::shared_ptr<NDTensorLiteral> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<NDTensorLiteral>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct ApplyStmt : public Stmt {
            UnreducedMapExpr::Ptr map;

            typedef std::shared_ptr<ApplyStmt> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ApplyStmt>());
            }

            virtual unsigned getLineBegin() { return map->getLineBegin(); }

            virtual unsigned getColBegin() { return map->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        struct Test : public FIRNode {
            Identifier::Ptr func;
            std::vector<Expr::Ptr> args;
            Expr::Ptr expected;

            typedef std::shared_ptr<Test> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Test>());
            }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct VertexSetType : public Type {
            ElementType::Ptr element;

            typedef std::shared_ptr<VertexSetType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<VertexSetType>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };


        struct EdgeSetType : public Type {
            ElementType::Ptr edge_element_type;
            std::vector<ElementType::Ptr> vertex_element_type_list;
            ScalarType::Ptr weight_type;
            typedef std::shared_ptr<EdgeSetType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<EdgeSetType>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };


        struct NewExpr : public Expr {
            typedef std::shared_ptr<NewExpr> Ptr;
            //A more general type(can be nested). For example, Vertex in list{vertexset{Vertex}}
            Type::Ptr general_element_type;

            //A simpler elementType, For example, vector{Vertex}(int)
            ElementType::Ptr elementType;
            Expr::Ptr numElements;
        };

        // Allocator expression for VertexSet
        struct VertexSetAllocExpr : public NewExpr {
            typedef std::shared_ptr<VertexSetAllocExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<VertexSetAllocExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };


        // Allocator expression for List
        struct ListAllocExpr : public NewExpr {
            typedef std::shared_ptr<ListAllocExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ListAllocExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };


        // Allocator expression for Vector
        struct VectorAllocExpr : public NewExpr {
            typedef std::shared_ptr<VectorAllocExpr> Ptr;
            fir::ScalarType::Ptr vector_scalar_type;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<VectorAllocExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();
            virtual void copy(FIRNode::Ptr);
        };

        struct IntersectionExpr : public Expr {
            typedef std::shared_ptr<IntersectionExpr> Ptr;
            Expr::Ptr vertex_a;
            Expr::Ptr vertex_b;
            Expr::Ptr numA;
            Expr::Ptr numB;
            Expr::Ptr reference;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<IntersectionExpr>());
            }

            protected:
                virtual FIRNode::Ptr cloneNode();

                virtual void copy(FIRNode::Ptr);
        };

        struct IntersectNeighborExpr : public Expr {
            typedef std::shared_ptr<IntersectNeighborExpr> Ptr;
            Expr::Ptr edges;
            Expr::Ptr vertex_a;
            Expr::Ptr vertex_b;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<IntersectNeighborExpr>());
            }

            protected:
               virtual FIRNode::Ptr cloneNode();

               virtual void copy(FIRNode::Ptr);
        };

        struct ConstantVectorExpr : public Expr {
            typedef std::shared_ptr<ConstantVectorExpr> Ptr;
            std::vector<Expr::Ptr> vectorElements;
            int numElements;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ConstantVectorExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };

        struct FuncExpr : public Expr {
            typedef std::shared_ptr<FuncExpr> Ptr;
            Identifier::Ptr name;
            std::vector<Expr::Ptr> args;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<FuncExpr>());
            }

            protected:
                virtual FIRNode::Ptr cloneNode();

                virtual void copy(FIRNode::Ptr);

        };

        struct LoadExpr : public Expr {
            typedef std::shared_ptr<LoadExpr> Ptr;
            //Currently unused for cleaner syntax
            //ElementType::Ptr element_type;
            Expr::Ptr file_name;
        };

        // Allocator expression for VertexSet
        struct EdgeSetLoadExpr : public LoadExpr {
            typedef std::shared_ptr<EdgeSetLoadExpr> Ptr;
            bool is_weighted = false;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<EdgeSetLoadExpr>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };

        struct MethodCallExpr : public Expr {
            Identifier::Ptr method_name;
            Expr::Ptr target;
            std::vector<Expr::Ptr> args;

            typedef std::shared_ptr<MethodCallExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<MethodCallExpr>());
            }

            virtual unsigned getLineBegin() { return method_name->getLineBegin(); }

            virtual unsigned getColBegin() { return method_name->getColBegin(); }

        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };



        struct WhereExpr : public Expr {
            Expr::Ptr target;
            FuncExpr::Ptr input_func;

            typedef std::shared_ptr<WhereExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<WhereExpr>());
            }


        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct FromExpr : public Expr {
            //Identifier::Ptr input_func;

            FuncExpr::Ptr input_func;

            typedef std::shared_ptr<FromExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<FromExpr>());
            }


        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct ToExpr : public Expr {
            //Identifier::Ptr input_func;

            FuncExpr::Ptr input_func;

            typedef std::shared_ptr<ToExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ToExpr>());
            }


        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };


        struct ApplyExpr : public Expr {

            enum class Type {
                REGULAR_APPLY, UPDATE_PRIORITY_APPLY, UPDATE_PRIORITY_EXTERN_APPLY
            };

            Expr::Ptr target;
            //Identifier::Ptr input_function;
            FuncExpr::Ptr input_function;
            FromExpr::Ptr from_expr;
            ToExpr::Ptr to_expr;
            Identifier::Ptr change_tracking_field;
            bool disable_deduplication = false;
            Type type = Type::REGULAR_APPLY;
            typedef std::shared_ptr<ApplyExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<ApplyExpr>());
            }


        protected:
            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode();
        };

        // OG additions

        struct PriorityQueueType : public Type {
            ElementType::Ptr element;
            ScalarType::Ptr priority_type;

            typedef std::shared_ptr<PriorityQueueType> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<PriorityQueueType>());
            }

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };

        struct PriorityQueueAllocExpr : public NewExpr {
            typedef std::shared_ptr<PriorityQueueAllocExpr> Ptr;

            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<PriorityQueueAllocExpr>());
            }

            Expr::Ptr dup_within_bucket;
            Expr::Ptr dup_across_bucket;
            Identifier::Ptr vector_function;
            Expr::Ptr bucket_ordering;
            Expr::Ptr priority_ordering;
            Expr::Ptr init_bucket;
            Expr::Ptr starting_node;

            ScalarType::Ptr priority_type;

        protected:
            virtual FIRNode::Ptr cloneNode();

            virtual void copy(FIRNode::Ptr);
        };

        // Utility functions
        typedef std::vector<IndexSet::Ptr> IndexDomain;
        typedef std::vector<IndexDomain> TensorDimensions;

        TensorType::Ptr
        makeTensorType(ScalarType::Type componentType,
                       const TensorDimensions &dimensions = TensorDimensions(),
                       bool transposed = false);
    }
}

#endif //GRAPHIT_FIR_H
