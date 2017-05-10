//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H

#include <string>

namespace graphit {

    struct PhysicalDataLayout {
        std::string var_name;
        std::string data_layout_type;
    };

    class Schedule {
    public:
        Schedule() {};
        //TODO: what does it mean??
        std::vector<PhysicalDataLayout> &physical_data_layouts();
    };
}


#endif //GRAPHIT_SCHEDULE_H
