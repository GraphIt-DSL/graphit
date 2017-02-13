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

            //void setProgram(mir::Stmt::Ptr program){this->mir_program = program};
            void addStatement(mir::Stmt::Ptr stmt){
                statements.push_back(stmt);
            }

        private:
            //mir::Program::Ptr mir_program;
            std::vector<mir::Stmt::Ptr> statements;
        };
    }
}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
