//
// Created by Yunming Zhang on 3/20/19.
//

#ifndef GRAPHIT_PRIORITY_FEATURES_LOWERING_H
#define GRAPHIT_PRIORITY_FEATURES_LOWERING_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>


namespace graphit {

    class PriorityFeaturesLower {
    public:

        PriorityFeaturesLower(MIRContext *mir_context) : mir_context_(mir_context) {};

        PriorityFeaturesLower(MIRContext *mir_context, Schedule *schedule)
                : schedule_(schedule), mir_context_(mir_context) {};


        struct PriorityUpdateScheduleFinder : public mir::MIRVisitor {
            using mir::MIRVisitor::visit;

            PriorityUpdateScheduleFinder(MIRContext *mir_context, Schedule *schedule)
                    : mir_context_(mir_context), schedule_(schedule) {

            };

            void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr update_priority_edgeset_apply_expr);

            void visit(mir::UpdatePriorityExternVertexSetApplyExpr::Ptr update_priority_extern_vertexset_apply_expr);

            void setPrioritySchedule(std::string current_label);

            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;

        };


        // use the schedule to set the type for MIR::PriorityQeueuType and PriorityQueueAllocExpr
        struct LowerPriorityRelatedTypeAndExpr : public mir::MIRVisitor {
            using mir::MIRVisitor::visit;

            LowerPriorityRelatedTypeAndExpr(MIRContext *mir_context, Schedule *schedule)
                    : mir_context_(mir_context), schedule_(schedule) {};

            void visit(mir::PriorityQueueType::Ptr priority_queue_type) {
                priority_queue_type->priority_update_type = mir_context_->priority_update_type;
            }

            void visit(mir::PriorityQueueAllocExpr::Ptr priority_queue_alloc_expr) {
                priority_queue_alloc_expr->priority_update_type = mir_context_->priority_update_type;
                priority_queue_alloc_expr->delta = mir_context_->delta_;
                mir_context_->optional_starting_source_node = priority_queue_alloc_expr->starting_node;
                mir_context_->priority_queue_alloc_list_.push_back(priority_queue_alloc_expr);
                mir_context_->nodes_init_in_buckets = priority_queue_alloc_expr->init_bucket;
            }

            void visit(mir::EdgeSetLoadExpr::Ptr edgeset_load_expr) {
                edgeset_load_expr->priority_update_type = mir_context_->priority_update_type;
            }

            void visit(mir::VertexSetType::Ptr vertex_set_type) {
                vertex_set_type->priority_update_type = mir_context_->priority_update_type;
            }

            void visit(mir::EdgeSetType::Ptr edge_set_type) {
                edge_set_type->priority_update_type = mir_context_->priority_update_type;
            }

            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;
        };

        struct LowerUpdatePriorityExternVertexSetApplyExpr : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerUpdatePriorityExternVertexSetApplyExpr(Schedule *schedule, MIRContext *mir_context) : schedule_(
                    schedule), mir_context_(mir_context) {

            }

            virtual void visit(mir::ExprStmt::Ptr);

            Schedule *schedule_;
            MIRContext *mir_context_;
        };

        // Do code lowering for reduce before priority update (the default schedule)
        struct LowerReduceBeforePriorityUpdate : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerReduceBeforePriorityUpdate(Schedule *schedule, MIRContext *mir_context) : schedule_(
                    schedule), mir_context_(mir_context) {

            }

            //specialize the dequeue_ready_set call for the GraphIt VertexSubset
            virtual void visit(mir::Call::Ptr);

            Schedule *schedule_;
            MIRContext *mir_context_;
        };


        struct LowerIntoOrderedProcessingOperatorRewriter : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerIntoOrderedProcessingOperatorRewriter(Schedule *schedule, MIRContext *mir_context) : schedule_(
                    schedule), mir_context_(mir_context) {

            }

            virtual void visit(mir::WhileStmt::Ptr while_stmt);

            bool checkWhileStmtPattern(mir::WhileStmt::Ptr while_stmt);

            Schedule *schedule_;
            MIRContext *mir_context_;

        };

        //lowers a few special call expressions that updates the priority
        struct LowerPriorityUpdateOperatorRewriter : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerPriorityUpdateOperatorRewriter(Schedule *schedule, MIRContext *mir_context) : schedule_(
                    schedule), mir_context_(mir_context) {

            }

            virtual void visit(mir::Call::Ptr call);


            Schedule *schedule_;
            MIRContext *mir_context_;

        };

        struct LowerUpdatePriorityEdgeSetApplyExpr : public mir::MIRRewriter {
            using mir::MIRRewriter::visit;

            LowerUpdatePriorityEdgeSetApplyExpr(Schedule *schedule, MIRContext *mir_context) :
                    schedule_(schedule), mir_context_(mir_context) {
            }

            //virtual void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr expr);
            virtual void visit(mir::ExprStmt::Ptr stmt);

            void lowerReduceBeforeUpdateEdgeSetApply(mir::ExprStmt::Ptr stmt);

            Schedule *schedule_;
            MIRContext *mir_context_;
        };


        void lower();


    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;

    };


}


#endif //GRAPHIT_PRIORITY_FEATURES_LOWERING_H
