//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/frontend/fir_printer.h>


namespace graphit {
    namespace fir {

        void FIRPrinter::visit(Program::Ptr program) {
            for (auto elem : program->elems) {
                elem->accept(this);
            }
            oss << std::endl;
        }

        void FIRPrinter::visit(Stmt::Ptr stmt) {
            stmt->accept(this);
        };

        void FIRPrinter::visit(Expr::Ptr expr) {
            expr->accept(this);
        };

        void FIRPrinter::visit(IntLiteral::Ptr lit) {
            oss << lit->val;
        }

        void FIRPrinter::visit(AddExpr::Ptr expr) {
            printBinaryExpr(expr, "+");
        }

        void FIRPrinter::visit(MinusExpr::Ptr expr) {
            printBinaryExpr(expr, "-");
        }

        void FIRPrinter::printBinaryExpr(BinaryExpr::Ptr expr, const std::string op) {
            oss << "(";
            expr->lhs->accept(this);
            oss << ") " << op << " (";
            expr->rhs->accept(this);
            oss << ")";
        }

        std::ostream &operator<<(std::ostream &oss, FIRNode &node) {
            FIRPrinter printer(oss);
            node.accept(&printer);
            return oss;
        }

    }
}