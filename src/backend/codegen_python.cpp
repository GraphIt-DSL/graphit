//
// Created by Ajay Brahmakshatriya on 5/14/2019
//

#include <graphit/backend/codegen_python.h>
#include <graphit/midend/mir.h>

namespace graphit {
    int CodeGenPython::genPython() {


	generatePythonImports();
        //Processing the functions
        std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();

	
        for (auto it = functions.begin(); it != functions.end(); it++) {
	        it->get()->accept(this);
        }

        oss << std::endl;
        return 0;
    }
    void CodeGenPython::generatePythonImports(void) {
	oss << "import scipy" << std::endl;
	oss << "import scipy.sparse" << std::endl;
	oss << "import sys" << std::endl;
	oss << "sys.path.insert(0, \"" << module_path << "\")" << std::endl;
	oss << "import " << module_name << "__imp" << std::endl;
    }
    void CodeGenPython::visit(mir::FuncDecl::Ptr func_decl) {
	if (func_decl->type == mir::FuncDecl::Type::EXPORTED) {
		oss << "def " << func_decl->name << "(";
		bool printDelimeter = false;
		for (auto arg: func_decl->args) {
			if (printDelimeter)
				oss << ", ";
			oss << arg.getName();
			printDelimeter = true;
		}
		oss << "):" << std::endl;
		indent();
		printIndent();
		// We are not returning Graph types right now, so no special handling of return types required.
		oss << "return " << module_name << "__imp.";
		oss << func_decl->name << "(";
		printDelimeter = false;
		for (auto arg : func_decl->args) {
			if(printDelimeter)
				oss << ", ";
			if (mir::isa<mir::EdgeSetType>(arg.getType())) {
				oss << arg.getName() << ".data, " << arg.getName() << ".indices, " << arg.getName() << ".indptr";
			}else{
				oss << arg.getName();
			}
		}
		oss << ")" << std::endl;
		dedent();
		oss << std::endl;

	}
    }
}
