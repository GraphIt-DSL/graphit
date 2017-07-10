//
// Created by Yunming Zhang on 7/10/17.
//

#ifndef GRAPHIT_GEN_EDGE_APPLY_FUNC_DECL_H
#define GRAPHIT_GEN_EDGE_APPLY_FUNC_DECL_H

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>
#include <iostream>
#include <sstream>

namespace graphit {

    /**
     * Generates function declarations for various edgeset apply operations with different schedules
     */
    struct GenEdgeApplyFunctionVisitor : mir::MIRVisitor {

        virtual void visit (mir::PushEdgeSetApplyExpr::Ptr push_apply);
        virtual void visit (mir::PullEdgeSetApplyExpr::Ptr pull_apply);

        GenEdgeApplyFunctionVisitor(MIRContext* mir_context) : mir_context_(mir_context){
        }



        void genEdgeApplyFuncDecls(){
            //Processing the functions
            std::map<std::string, mir::FuncDecl::Ptr>::iterator it;
            std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
            for (auto it = functions.begin(); it != functions.end(); it++) {
                it->get()->accept(this);
            }
        }

        std::string genFunctionName(mir::EdgeSetApplyExpr::Ptr push_apply);

    private:
        MIRContext* mir_context_;
    };
}

#endif //GRAPHIT_GEN_EDGE_APPLY_FUNC_DECL_H
