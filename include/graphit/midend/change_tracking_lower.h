//
// Created by Yunming Zhang on 6/29/17.
//

#ifndef GRAPHIT_CHANGE_TRACKING_LOWER_H
#define GRAPHIT_CHANGE_TRACKING_LOWER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {

    class ChangeTrackingLower {
    public:
        ChangeTrackingLower(MIRContext *mir_context, Schedule *schedule)
                : schedule_(schedule), mir_context_(mir_context) {};

        void lower();

        struct TrackingVariableGenVisitor : public mir::MIRVisitor {
            TrackingVariableGenVisitor(MIRContext* mir_context)
                    : mir_context_(mir_context){
                field_vector_variable_map_ = std::unordered_map<std::string, std::vector<std::string>>();
            }

            virtual void visit(mir::AssignStmt::Ptr assign_stmt);
            virtual void visit(mir::ReduceStmt::Ptr reduce_stmt);
            virtual void visit(mir::CompareAndSwapStmt::Ptr assign_stmt);
            virtual void visit(mir::ExprStmt::Ptr);

            mir::Expr::Ptr getFieldTrackingVariableExpr(std::string field_vector_name);

        private:
            std::unordered_map<std::string, std::vector<std::string>> field_vector_variable_map_ ;
            MIRContext *mir_context_;

            void addFieldTrackingVariable(std::string field_vector_name, std::string tracking_var);

        };

        struct ApplyExprVisitor : public mir::MIRVisitor {

            ApplyExprVisitor(MIRContext *mir_context, Schedule *schedule) :
                    mir_context_(mir_context), schedule_(schedule){}

            virtual void visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::HybridDenseForwardEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr);


        private:
            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;

            void
            insertSerialReturnStmtForTrackingChange(mir::FuncDecl::Ptr apply_func_decl,
                                                                std::string tracking_field,
                                                                mir::Expr::Ptr tracking_var);

            void processSingleFunctionApplyExpr(std::string apply_func_decl_name,
                                                            std::string tracking_field);
        };

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;

    };
}

#endif //GRAPHIT_CHANGE_TRACKING_LOWER_H
