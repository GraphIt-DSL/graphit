//
// Created by Yunming Zhang on 1/24/17.
//

#ifndef GRAPHIT_FIR_VISITOR_H
#define GRAPHIT_FIR_VISITOR_H

#endif //GRAPHIT_FIR_VISITOR_H
#include <memory>

namespace graphit {
    namespace fir {

        struct Program;



        struct FIRVisitor {
            virtual void visit(std::shared_ptr<Program>);
        };
    }

}