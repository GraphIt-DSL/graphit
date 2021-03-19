#include "graphit/midend/udf_dup.h"

namespace graphit {


static void duplicate_udf(mir::EdgeSetApplyExpr::Ptr esae, MIRContext *mir_context_) {
    auto func_decl = mir_context_->getFunction(esae->input_function->function_name->name);
    mir::FuncDecl::Ptr new_func_decl = func_decl->clone<mir::FuncDecl>();
    new_func_decl->name = new_func_decl->name + mir_context_->getUniqueNameCounterString(); 
    
    esae->input_function->function_name->name = new_func_decl->name; 
    mir_context_->addFunctionFront(new_func_decl);
     
} 

void UDFReuseFinder::lower(void) {	
    std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
    for (auto function : functions) {
        function->accept(this);
    }
    
    for (auto iter = udf_usage_map.begin(); iter != udf_usage_map.end(); iter++) {
        std::string fname = iter->first;
        if (udf_usage_map[fname].size() > 1) {
            for (int i = 1; i < udf_usage_map[fname].size(); i++) {
                duplicate_udf(udf_usage_map[fname][i], mir_context_);
            }
        }
        
    }    
}

void UDFReuseFinder::visit(mir::EdgeSetApplyExpr::Ptr esae) {
    std::string fname = esae->input_function->function_name->name;
    udf_usage_map[fname].push_back(esae);	

}

}
