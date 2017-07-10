//
// Created by Yunming Zhang on 7/10/17.

#include <graphit/backend/gen_edge_apply_func_decl.h>

namespace graphit {


    void GenEdgeApplyFunctionVisitor::visit(mir::PushEdgeSetApplyExpr::Ptr push_apply) {

    }

    void GenEdgeApplyFunctionVisitor::visit(mir::PullEdgeSetApplyExpr::Ptr pull_apply) {

    }

    std::string GenEdgeApplyFunctionVisitor::genFunctionName(mir::EdgeSetApplyExpr::Ptr push_apply) {
        return "";
    }
}