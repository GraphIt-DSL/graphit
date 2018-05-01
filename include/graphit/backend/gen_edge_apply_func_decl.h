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
#include <string>

namespace graphit {

    /**
     * Generates function declarations for various edgeset apply operations with different schedules
     */
    struct EdgesetApplyFunctionDeclGenerator : mir::MIRVisitor {

        virtual void visit (mir::PushEdgeSetApplyExpr::Ptr push_apply);
        virtual void visit (mir::PullEdgeSetApplyExpr::Ptr pull_apply);
        virtual void visit (mir::HybridDenseEdgeSetApplyExpr::Ptr hybrid_dense_apply);
        virtual void visit (mir::HybridDenseForwardEdgeSetApplyExpr::Ptr hybrid_dense_forward_apply);

        EdgesetApplyFunctionDeclGenerator(MIRContext* mir_context, std::ostream& oss)
                : mir_context_(mir_context), oss_ (oss){
            indentLevel = 0;
        }



        void genEdgeApplyFuncDecls(){
            //Processing the functions
            std::map<std::string, mir::FuncDecl::Ptr>::iterator it;
            std::vector<mir::FuncDecl::Ptr> functions = mir_context_->getFunctionList();
            for (auto it = functions.begin(); it != functions.end(); it++) {
                it->get()->accept(this);
            }
        }

        // figure out the right function name for the particular edgeset apply function
        std::string genFunctionName(mir::EdgeSetApplyExpr::Ptr push_apply);


    private:
        MIRContext* mir_context_;
        std::ostream &oss_;

        void genEdgeApplyFunctionSignature(mir::EdgeSetApplyExpr::Ptr apply);
        void genEdgeApplyFunctionDeclaration(mir::EdgeSetApplyExpr::Ptr apply);
        void genEdgeApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply);

        void indent() { ++indentLevel; }
        void dedent() { --indentLevel; }
        void printIndent() { oss_ << std::string(2 * indentLevel, ' '); }
        void printBeginIndent() { oss_ << std::string(2 * indentLevel, ' ') << "{" << std::endl; }
        void printEndIndent() { oss_ << std::string(2 * indentLevel, ' ') << "}"; }
        unsigned      indentLevel;

        void genEdgePullApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply);
        void genEdgePushApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply);
        void genEdgeHybridDenseApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply);
        void genEdgeHybridDenseForwardApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply);
        void setupGlobalVariables(mir::EdgeSetApplyExpr::Ptr apply,
                                  bool apply_expr_gen_frontier,
                                  bool from_vertexset_specified);
        void setupFlags(mir::EdgeSetApplyExpr::Ptr apply,
                        bool & from_vertexset_specified,
                        bool & apply_expr_gen_frontier,
                        std::string & dst_type);
        void printPushEdgeTraversalReturnFrontier(mir::EdgeSetApplyExpr::Ptr apply,
                                          bool from_vertexset_specified,
                                          bool apply_expr_gen_frontier,
                                          std::string dst_type,
                                          std::string apply_func_name = "apply_func");

        void printPullEdgeTraversalReturnFrontier(mir::EdgeSetApplyExpr::Ptr apply,
                                                  bool from_vertexset_specified,
                                                  bool apply_expr_gen_frontier,
                                                  std::string dst_type,
                                                  std::string apply_func_name = "apply_func");
        void printHybridDenseEdgeTraversalReturnFrontier(mir::EdgeSetApplyExpr::Ptr apply,
                                                  bool from_vertexset_specified,
                                                  bool apply_expr_gen_frontier,
                                                  std::string dst_type);

        void printHybridDenseForwardEdgeTraversalReturnFrontier(mir::EdgeSetApplyExpr::Ptr apply,
                                                         bool from_vertexset_specified,
                                                         bool apply_expr_gen_frontier,
                                                         std::string dst_type);

        void printDenseForwardEdgeTraversalReturnFrontier(mir::EdgeSetApplyExpr::Ptr apply,
                                                                bool from_vertexset_specified,
                                                                bool apply_expr_gen_frontier,
                                                                std::string dst_type);

        //prints the inner loop on in neighbors for pull based direction
        void printPullEdgeTraversalInnerNeighborLoop(mir::EdgeSetApplyExpr::Ptr apply,
                                                     bool from_vertexset_specified,
                                                     bool apply_expr_gen_frontier,
                                                     std::string dst_type,
                                                     std::string apply_func_name,
                                                     bool cache,
                                                     bool numa_aware);

        void printNumaMerge(mir::EdgeSetApplyExpr::Ptr apply);

        void printNumaScatter(mir::EdgeSetApplyExpr::Ptr apply);

    };
}

#endif //GRAPHIT_GEN_EDGE_APPLY_FUNC_DECL_H
