//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H

#include <string>
#include <map>

namespace graphit {

    //TODO: move this into the PhysicalDataLayout class definition
    /** An enum describing a type of physical data layout */
    enum class DataLayoutType {
        ARRAY,
        DICT,
        STRUCT
    };



    struct PhysicalDataLayout {
        std::string var_name;
        DataLayoutType data_layout_type;
        // Records the name of the struct for all fused fields
        std::string fused_struct_name;
    };

    struct ApplySchedule {
        enum class DirectionType {
            PUSH,
            PULL,
            HYBRID
        };
        std::string scope_label_name;
        DirectionType direction_type;
    };

    /**
     * User specified schedule object
     */
    class Schedule {
    public:
        Schedule() {
            physical_data_layouts = new std::map<std::string, PhysicalDataLayout>();
            apply_schedules = new std::map<std::string, ApplySchedule>();
        };

        ~Schedule(){
            delete physical_data_layouts;
            delete apply_schedules;
        }

        //TODO: what does it mean??
        std::map<std::string, PhysicalDataLayout> *physical_data_layouts;
        std::map<std::string, ApplySchedule>* apply_schedules;
    };
}


#endif //GRAPHIT_SCHEDULE_H
