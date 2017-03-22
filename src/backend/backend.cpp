//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/backend.h>

namespace graphit{
    int Backend::emitCPP() {
        CodeGenCPP* codegen_cpp = new CodeGenCPP(std::cout);
        int flag = codegen_cpp->genCPP(mir_context_);
        delete codegen_cpp;
        return flag;
    }
}