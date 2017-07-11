//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H

#include <string>
#include <map>

namespace graphit {

    //TODO: move this into the FieldVectorPhysicalDataLayout class definition
    /** An enum describing a type of physical data layout */
    enum class FieldVectorDataLayoutType {
        ARRAY,
        DICT,
        STRUCT
    };



    struct FieldVectorPhysicalDataLayout {
        std::string var_name;
        FieldVectorDataLayoutType data_layout_type;
        // Records the name of the struct for all fused fields
        std::string fused_struct_name;
    };

    struct VertexsetPhysicalLayout {
        enum class DataLayout {
            SPARSE,
            DENSE
        };
        std::string vertexset_label;
        DataLayout data_layout_type;
    };

    struct ApplySchedule {
        enum class DirectionType {
            PUSH,
            PULL,
            HYBRID
        };

        enum class ParType {
            Parallel,
            Serial
        };

        enum class FrontierType{
            Sparse,
            Dense
        };

        enum class DeduplicationType {
            Enable,
            Disable
        };

        std::string scope_label_name;
        DirectionType direction_type;
        ParType parallel_type;
        //FrontierType frontier_type;
        DeduplicationType deduplication_type;
    };

    /**
     * User specified schedule object
     */
    class Schedule {
    public:
        Schedule() {
            physical_data_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
            apply_schedules = new std::map<std::string, ApplySchedule>();
            vertexset_data_layout =  std::map<std::string, VertexsetPhysicalLayout>();
        };

        ~Schedule(){
            delete physical_data_layouts;
            delete apply_schedules;
        }

        //TODO: what does it mean??
        std::map<std::string, FieldVectorPhysicalDataLayout> *physical_data_layouts;
        std::map<std::string, ApplySchedule>* apply_schedules;
        std::map<std::string, VertexsetPhysicalLayout> vertexset_data_layout;
    };
}


#endif //GRAPHIT_SCHEDULE_H
