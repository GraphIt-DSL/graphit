//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/backend.h>

namespace graphit{
    int Backend::emitCPP(std::ostream &oss, std::string module_name) {
        CodeGenCPP* codegen_cpp = new CodeGenCPP(oss, mir_context_, module_name);
        int flag = codegen_cpp->genCPP();
        delete codegen_cpp;
        return flag;
    }
    int Backend::emitPython(std::ostream &oss, std::string module_name, std::string module_path) {
	CodeGenPython *codegen_python = new CodeGenPython(oss, mir_context_, module_name, module_path);
	int flag = codegen_python->genPython();
	delete codegen_python;
	return flag;
    }
    int Backend::emitGPU(std::ostream &oss, std::string module_name, std::string module_path) {
        CodeGenGPU code_gen_gpu(oss, mir_context_, module_name, module_path);
	return code_gen_gpu.genGPU();
    }
}
