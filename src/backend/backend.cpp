//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/backend.h>
#include <graphit/backend/codegen_swarm.h>

namespace graphit{
    int Backend::emitCPP(std::ostream &oss, std::string module_name) {
	return CodeGenSwarm(oss, mir_context_, module_name).genSwarmCode();
    }
    int Backend::emitPython(std::ostream &oss, std::string module_name, std::string module_path) {
	CodeGenPython *codegen_python = new CodeGenPython(oss, mir_context_, module_name, module_path);
	int flag = codegen_python->genPython();
	delete codegen_python;
	return flag;
    }
}
