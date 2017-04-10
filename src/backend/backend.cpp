//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/backend.h>

namespace graphit{
    int Backend::emitCPP(std::ostream &oss) {
        CodeGenCPP* codegen_cpp = new CodeGenCPP(oss);
        int flag = codegen_cpp->genCPP(mir_context_);
        delete codegen_cpp;
        return flag;
    }
}