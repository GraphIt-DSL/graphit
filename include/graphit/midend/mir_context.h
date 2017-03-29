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


        // Data structure that holds the internal representation of the program
        class MIRContext {

        public:
            MIRContext() {
            }


            ~MIRContext() {
            }

            //void setProgram(mir::Stmt::Ptr program){this->mir_program = program};
            void addConstant(mir::VarDecl::Ptr var_decl){
                constants.push_back(var_decl);
            }

            std::vector<mir::VarDecl::Ptr> getConstants(){
                return constants;
            }

        private:
            //mir::Program::Ptr mir_program;
            std::vector<mir::VarDecl::Ptr> constants;
        };

}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
