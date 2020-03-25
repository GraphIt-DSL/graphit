//
// Created by Yunming Zhang on 2/9/17.
//

#ifndef GRAPHIT_FIR_PRINTER_H
#define GRAPHIT_FIR_PRINTER_H

#include <iostream>

#include "fir.h"
#include "fir_visitor.h"

// prints FIR using visitor pattern. Overloads the stream (<<) operator of FIRNode in fir_printer.cpp
namespace graphit {
    namespace fir{
        struct FIRPrinter : public FIRVisitor {
            FIRPrinter(std::ostream &oss) : oss(oss), indentLevel(0) {}

        protected:
            virtual void visit(Program::Ptr);
            virtual void visit(StmtBlock::Ptr);
            virtual void visit(RangeIndexSet::Ptr);
            virtual void visit(SetIndexSet::Ptr);
            virtual void visit(DynamicIndexSet::Ptr);
            virtual void visit(ElementType::Ptr);
            virtual void visit(Endpoint::Ptr);
            virtual void visit(HomogeneousEdgeSetType::Ptr);
            virtual void visit(HeterogeneousEdgeSetType::Ptr);
            virtual void visit(GridSetType::Ptr);
            virtual void visit(TupleElement::Ptr);
            virtual void visit(NamedTupleType::Ptr);
            virtual void visit(TupleLength::Ptr);
            virtual void visit(UnnamedTupleType::Ptr);
            virtual void visit(ScalarType::Ptr);
            virtual void visit(NDTensorType::Ptr);
            virtual void visit(OpaqueType::Ptr);
            virtual void visit(Identifier::Ptr);
            virtual void visit(IdentDecl::Ptr);
            virtual void visit(FieldDecl::Ptr);
            virtual void visit(ElementTypeDecl::Ptr);
            virtual void visit(Argument::Ptr);
            virtual void visit(ExternDecl::Ptr);
            virtual void visit(GenericParam::Ptr);
            virtual void visit(FuncDecl::Ptr);
            virtual void visit(VarDecl::Ptr);
            virtual void visit(ConstDecl::Ptr);
            virtual void visit(WhileStmt::Ptr);
            virtual void visit(DoWhileStmt::Ptr);
            virtual void visit(IfStmt::Ptr);
            virtual void visit(IndexSetDomain::Ptr);
            virtual void visit(RangeDomain::Ptr);
            virtual void visit(ForStmt::Ptr);
            virtual void visit(PrintStmt::Ptr);

            virtual void visit(BreakStmt::Ptr);

            virtual void visit(ExprStmt::Ptr);
            virtual void visit(AssignStmt::Ptr);
            virtual void visit(ReduceStmt::Ptr);


            virtual void visit(Slice::Ptr);
            virtual void visit(ExprParam::Ptr);
            virtual void visit(MapExpr::Ptr);
            virtual void visit(OrExpr::Ptr);
            virtual void visit(AndExpr::Ptr);
            virtual void visit(XorExpr::Ptr);
            virtual void visit(EqExpr::Ptr);
            virtual void visit(NotExpr::Ptr);
            virtual void visit(AddExpr::Ptr);
            virtual void visit(SubExpr::Ptr);
            virtual void visit(MulExpr::Ptr);
            virtual void visit(DivExpr::Ptr);
            virtual void visit(LeftDivExpr::Ptr);
            virtual void visit(ElwiseMulExpr::Ptr);
            virtual void visit(ElwiseDivExpr::Ptr);
            virtual void visit(NegExpr::Ptr);
            virtual void visit(ExpExpr::Ptr);
            virtual void visit(TransposeExpr::Ptr);
            virtual void visit(CallExpr::Ptr);
            virtual void visit(TensorReadExpr::Ptr);
            virtual void visit(SetReadExpr::Ptr);
            virtual void visit(NamedTupleReadExpr::Ptr);
            virtual void visit(UnnamedTupleReadExpr::Ptr);
            virtual void visit(FieldReadExpr::Ptr);
            virtual void visit(VarExpr::Ptr);
            virtual void visit(IntLiteral::Ptr);
            virtual void visit(FloatLiteral::Ptr);
            virtual void visit(BoolLiteral::Ptr);
            virtual void visit(IntVectorLiteral::Ptr);
            virtual void visit(FloatVectorLiteral::Ptr);
            virtual void visit(NDTensorLiteral::Ptr);
            virtual void visit(ApplyStmt::Ptr);
            virtual void visit(Test::Ptr);

            virtual void visit(VertexSetType::Ptr);
            virtual void visit(ListType::Ptr);
            virtual void visit(VertexSetAllocExpr::Ptr);
            virtual void visit(ListAllocExpr::Ptr);
            virtual void visit(VectorAllocExpr::Ptr);

            virtual void visit(PriorityQueueType::Ptr);
            virtual void visit(PriorityQueueAllocExpr::Ptr);

            virtual void visit(FuncExpr::Ptr);
            virtual void visit(MethodCallExpr::Ptr);
            virtual void visit(ApplyExpr::Ptr);
            virtual void visit(WhereExpr::Ptr);
            virtual void visit(IntersectionExpr::Ptr);
            virtual void visit(IntersectNeighborExpr::Ptr);
            virtual void visit(EdgeSetLoadExpr::Ptr);
            virtual void visit(StringLiteral::Ptr);

            virtual void visit(NameNode::Ptr);

            void indent() { ++indentLevel; }
            void dedent() { --indentLevel; }
            void printIndent() { oss << std::string(2 * indentLevel, ' '); }
            void printBoolean(bool val) { oss << (val ? "true" : "false"); }

            void printIdentDecl(IdentDecl::Ptr);
            void printVarOrConstDecl(VarDecl::Ptr, const bool = false);
            void printMapOrApply(MapExpr::Ptr, const bool = false);
            void printUnaryExpr(UnaryExpr::Ptr, const std::string, const bool = false);
            void printBinaryExpr(BinaryExpr::Ptr, const std::string);

            std::ostream &oss;
            unsigned      indentLevel;
        };

    }
}

#endif //GRAPHIT_FIR_PRINTER_H