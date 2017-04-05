//
// Created by Yunming Zhang on 4/4/17.
//

#ifndef GRAPHIT_VAR_H
#define GRAPHIT_VAR_H
#include <graphit/midend/mir.h>

namespace graphit {
    namespace mir {


        class Var {
            //Type::Ptr type;
            std::string name_;

        public:
            //TODO: figure out where is this constructor used in scoped map
            Var(){};
            Var(std::string name) : name_(name){};
            //typedef std::shared_ptr<Var> Ptr;
            std::string getName(){
                return name_;
            }
        };

    }
}


#endif //GRAPHIT_VAR_H
