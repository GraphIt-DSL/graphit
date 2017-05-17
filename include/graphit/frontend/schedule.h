//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H

#include <string>
#include <map>

namespace graphit {

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

    /**
     * User specified schedule object
     */
    class Schedule {
    public:
        Schedule() {};
        //TODO: what does it mean??
        std::map<std::string, PhysicalDataLayout> *physical_data_layouts
                = new std::map<std::string, PhysicalDataLayout>();
    };
}


#endif //GRAPHIT_SCHEDULE_H
