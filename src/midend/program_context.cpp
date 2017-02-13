//
// Created by Yunming Zhang on 2/12/17.
//
#include <graphit/midend/program_context.h>

namespace graphit {
    namespace internal {


        void ProgramContext::addStatement(mir::Stmt::Ptr stmt){
            statements.front().push_back(stmt);
        };

    }
}