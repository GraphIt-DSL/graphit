//
// Created by Yunming Zhang on 2/14/17.
//

#ifndef GRAPHIT_BACKEND_H
#define GRAPHIT_BACKEND_H

#include <graphit/midend/mir_context.h>
#include <graphit/backend/codegen_cpp.h>
#include <graphit/backend/codegen_python.h>

namespace graphit {
    class Backend {
    public:
        Backend(MIRContext* mir_context) : mir_context_(mir_context){

        }

        int emitCPP(std::ostream &oss = std::cout, std::string module_name="");
    	int emitPython(std::ostream &oss = std::cout, std::string module_name="", std::string module_path="");

    private:
        MIRContext* mir_context_;
    };
}
#endif //GRAPHIT_BACKEND_H
