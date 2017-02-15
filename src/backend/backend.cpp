//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/backend.h>

namespace graphit{
    int Backend::emitCPP() {
        CodeGenCPP* codegen_cpp = new CodeGenCPP();
        return codegen_cpp->genCPP(mir_context_);

    }
}