//
// Created by Yunming Zhang on 2/13/17.
//
#include <graphit/midend/mir_printer.h>

namespace graphit {
    namespace mir {

//        void MIRPrinter::visit(VarDecl::Ptr expr) {
//            expr->accept(this);
//        };


//        void MIRPrinter::visit(FuncDecl::Ptr func_decl) {
//            oss << "func ";
//            oss << func_decl->name << " ";
//
//            oss << "(";
//
//            bool printDelimiter = false;
//            for (auto arg : func_decl->args) {
//                if (printDelimiter) {
//                    oss << ", ";
//                }
//
//                arg->accept(this);
//                printDelimiter = true;
//            }
//
//            oss << ") ";
//
//            if (!func_decl->result) {
//
//            }
//        };

        void MIRPrinter::visit(Expr::Ptr expr) {
            expr->accept(this);
        };

        void MIRPrinter::visit(IntLiteral::Ptr lit) {
            oss << lit->val;
        }

        void MIRPrinter::visit(AddExpr::Ptr expr) {
            printBinaryExpr(expr, "+");
        }

        void MIRPrinter::visit(SubExpr::Ptr expr) {
            printBinaryExpr(expr, "-");
        }

        void MIRPrinter::printBinaryExpr(BinaryExpr::Ptr expr, const std::string op) {
            oss << "(";
            expr->lhs->accept(this);
            oss << ") " << op << " (";
            expr->rhs->accept(this);
            oss << ")";
        }

        std::ostream &operator<<(std::ostream &oss, MIRNode &node) {
            MIRPrinter printer(oss);
            node.accept(&printer);
            return oss;
        }

    }
}