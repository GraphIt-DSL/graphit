//
// Created by Yunming Zhang on 2/12/17.
//

#ifndef GRAPHIT_PROGRAM_CONTEXT_H
#define GRAPHIT_PROGRAM_CONTEXT_H

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <utility>

#include <graphit/midend/mir.h>

namespace graphit {
    namespace internal {

        // Data structure that holds the internal representation of the program
        class ProgramContext {

        public:
            ProgramContext() {

            }


            ~ProgramContext() {

            }


            mir::Program::Ptr mid_ir;

        };
    }
}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
