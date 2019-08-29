//
// Created by Yunming Zhang on 6/29/17.
//

#ifndef GRAPHIT_VECTOR_FIELD_PROPERTIES_ANALYZER_H
#define GRAPHIT_VECTOR_FIELD_PROPERTIES_ANALYZER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/field_vector_property.h>

namespace graphit {


    /**
     * Analyze the function declarations used in apply operators
     * and figure out the properties of the field vectors used in the func body
     * This analyzer is useful for later insertion of atomics and change tracking code
     * Note, this analysis only makes sense for functions used for apply operators, and not the generarl functions
     * because the analysis on shared and local requires info on the direction
     */
    class VectorFieldPropertiesAnalyzer {
    public:
        VectorFieldPropertiesAnalyzer(MIRContext *mir_context, Schedule *schedule)
                : schedule_(schedule), mir_context_(mir_context) {};

        // the main analyze method
        void analyze();

        struct PropertyAnalyzingVisitor : public mir::MIRVisitor {

            PropertyAnalyzingVisitor(std::string direction, MIRContext* mir_context)
            : direction_(direction), mir_context_(mir_context) {
                in_write_phase = false;
                in_read_phase = false;
                in_read_write_phase = false;
            };

            virtual void visit(mir::AssignStmt::Ptr);

            virtual void visit(mir::ReduceStmt::Ptr);

            virtual void visit(mir::TensorReadExpr::Ptr);

            virtual void visit(mir::PriorityUpdateOperatorSum::Ptr);

            virtual void visit(mir::PriorityUpdateOperatorMin::Ptr);


        private:
            std::string direction_;
            MIRContext * mir_context_;

        private:
            bool in_write_phase;
            bool in_read_phase;
            bool in_read_write_phase;

            FieldVectorProperty buildLocalWriteFieldProperty();

            FieldVectorProperty buildSharedWriteFieldProperty();

            FieldVectorProperty buildLocalReadFieldProperty();

            FieldVectorProperty buildSharedReadFieldProperty();

            FieldVectorProperty determineFieldVectorProperty(std::string field_vector_name,
                                                             bool in_write_phase,
                                                             bool in_read_phase,
                                                             std::string index,
                                                             std::string direction);

            FieldVectorProperty buildLocalReadWriteFieldProperty();

            FieldVectorProperty buildSharedReadWriteFieldProperty();
        };

        struct ApplyExprVisitor : public mir::MIRVisitor {

            ApplyExprVisitor(MIRContext *mir_context, Schedule *schedule) :
                    mir_context_(mir_context), schedule_(schedule) {}

            virtual void visit(mir::PullEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::PushEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::HybridDenseForwardEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::HybridDenseEdgeSetApplyExpr::Ptr apply_expr);

            virtual void visit(mir::UpdatePriorityEdgeSetApplyExpr::Ptr priority_update_expr);

        private:
            Schedule *schedule_ = nullptr;
            MIRContext *mir_context_ = nullptr;

            void analyzeSingleFunctionEdgesetApplyExpr(std::string function_name, std::string direction);
        };

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;
    };


}

#endif //GRAPHIT_VECTOR_FIELD_PROPERTIES_ANALYZER_H
