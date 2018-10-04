//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/backend.h>
#include <graphit/backend/codegen_cpp/codegen_cpp.h>

namespace graphit{
int Backend::emit(std::ostream &oss) {
		int flag;
		CodeGenCPP *codegen_cpp;
		
	switch(mir_context_->backend_selection) {
		case backend_cpp:
			codegen_cpp = new CodeGenCPP(oss, mir_context_);
			flag = codegen_cpp->genCPP();
			delete codegen_cpp;
			return flag;
			break;
		case backend_gpu:
			std::cerr << "Backend GPU not implemented, failing with error\n";
			return -1;
			break;
		default:
			std::cerr << "Invalid backend chosen, failing with error\n";
			return -1;
			break;
	}
}
}
