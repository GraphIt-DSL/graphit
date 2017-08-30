//
// Created by Yunming Zhang on 7/10/17.

#include <graphit/backend/gen_edge_apply_func_decl.h>

namespace graphit {
    using namespace std;

    void EdgesetApplyFunctionDeclGenerator::visit(mir::PushEdgeSetApplyExpr::Ptr push_apply) {
        genEdgeApplyFunctionDeclaration(push_apply);
    }

    void EdgesetApplyFunctionDeclGenerator::visit(mir::PullEdgeSetApplyExpr::Ptr pull_apply) {
        genEdgeApplyFunctionDeclaration(pull_apply);
    }


    void EdgesetApplyFunctionDeclGenerator::genEdgeApplyFunctionDeclaration(mir::EdgeSetApplyExpr::Ptr apply){
        auto func_name = genFunctionName(apply);
        // currently, we only generate code for the following schedules
        if (func_name != "edgeset_apply_pull_parallel"
                && func_name != "edgeset_apply_pull_parallel_from_vertexset_to_filter_func_with_frontier"
                && func_name != "edgeset_apply_pull_parallel_deduplicatied_from_vertexset_with_frontier"
                && func_name != "edgeset_apply_pull_parallel_weighted_deduplicatied_from_vertexset_with_frontier"
                ){
            return;
        }

        genEdgeApplyFunctionSignature(apply);
        oss_ << "{ " << endl; //the end of the function declaration
        genEdgeApplyFunctionDeclBody(apply);
        oss_ << "} " << endl; //the end of the function declaration

    }

    void EdgesetApplyFunctionDeclGenerator::genEdgeApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply) {
        if (mir::isa<mir::PullEdgeSetApplyExpr>(apply)){
            genEdgePullApplyFunctionDeclBody(apply);
        }
    }

    void EdgesetApplyFunctionDeclGenerator::genEdgePullApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply){

        bool apply_expr_gen_frontier = false;
        // Check if the apply function has a return value
        auto apply_func = mir_context_->getFunction(apply->input_function_name);


        // If apply function has a return value, then we need to return a temporary vertexsubset
        if (apply_func->result.isInitialized()) {
            // build an empty vertex subset if apply function returns
            apply_expr_gen_frontier = true;
            oss_ <<
            "  VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);\n"
            "  long numVertices = g.num_nodes(), numEdges = g.num_edges();\n"
            "  long m = from_vertexset->size();\n"
            "  bool * next = newA(bool, g.num_nodes());\n"
            "  parallel_for (int i = 0; i < numVertices; i++)next[i] = 0;\n";
        }

        indent();


        if (apply->from_func != "") {
            if (!mir_context_->isFunction(apply->from_func)) {
                printIndent();
                oss_ << "from_vertexset->toDense();" << std::endl;
            }
        }

        printIndent();

        std::string for_type = "for";
        if (apply->is_parallel)
            for_type = "parallel_for";

        std::string node_id_type = "NodeID";
        if (apply->is_weighted) node_id_type = "WNode";

        oss_ << for_type <<  " ( NodeID d=0; d < g.num_nodes(); d++) {" << std::endl;
        indent();

        if (apply->to_func != "") {
            printIndent();
            oss_ << "if (to_func(d)){ " << std::endl;
            indent();
        }

        printIndent();

        oss_ << "for(" << node_id_type << " s : g.in_neigh(d)){" << std::endl;


        // print the checks on filtering on sources s
        if (apply->from_func != "") {
            indent();
            printIndent();

            oss_ << "if";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply->from_func)) {
                //if the input expression is a function call
                oss_ << " (from_func(s)";

            } else {
                //the input expression is a vertex subset
                oss_ << " (from_vertexset->bool_map_[s] ";
            }
            oss_ << ") { " << std::endl;
        }

        indent();
        printIndent();
        if (apply_expr_gen_frontier) {
            oss_ << "if( ";
        }

        // generating the C++ code for the apply function call
        if (apply->is_weighted) {
            oss_ <<  "apply_func ( s.v , d, s.w )";
        } else {
            oss_ << "apply_func ( s , d  )";

        }

        if (!apply_expr_gen_frontier) {
            // no need to generate a frontier
            oss_ << ";" << std::endl;
        } else {
            indent();
            //generate the code for adding destination to "next" frontier
            //TODO: fix later
            oss_ << " ) { "<< std::endl;
            printIndent();
            oss_ << "next[d] = 1; " << std::endl;
            // generating code for early break
            if (apply->to_func != "") {
                printIndent();
                oss_ << "if (!to_func(d)) break; " << std::endl;
            }
            dedent();
            printIndent();
            oss_ << "}" << std::endl;
        }



        // end of from filtering
        if (apply->from_func != "") {
            dedent();
            printIndent();
            oss_ << "}" << std::endl;
        }

        //end of for loop on the neighbors
        dedent();
        printIndent();
        oss_ << "}" << std::endl;

        if (apply->to_func != "") {
            dedent();
            printIndent();
            oss_ << "} " << std::endl;
        }


        dedent();
        printIndent();
        oss_ << "}" << std::endl;

        //return a new vertexset if no subset vertexset is returned
        if (!apply_expr_gen_frontier){
            printIndent();
            oss_ << "return new VertexSubset<NodeID>(g.num_nodes(), g.num_nodes());" << std::endl;
        } else {
            oss_ << "  next_frontier->num_vertices_ = sequence::sum(next, numVertices);\n"
                    "  next_frontier->bool_map_ = next;\n"
                    "  return next_frontier;\n";
        }

    }

    void EdgesetApplyFunctionDeclGenerator::genEdgeApplyFunctionSignature(mir::EdgeSetApplyExpr::Ptr apply){
        auto func_name = genFunctionName(apply);

        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(apply->target);
        vector<string> templates = vector<string>();
        vector<string> arguments = vector<string>();

        if (apply->is_weighted){
            arguments.push_back("WGraph & g");
        }else{
            arguments.push_back("Graph & g");
        }

        if (apply->from_func != ""){
            if (mir_context_->isFunction(apply->from_func)){
                // the schedule is an input from function
                templates.push_back("typename FROM_FUNC");
                arguments.push_back("FROM_FUNC from_func");
            } else {
                // the input is an input from vertexset
                arguments.push_back("VertexSubset<NodeID>* from_vertexset");
            }
        }

        if (apply->to_func != ""){
            if (mir_context_->isFunction(apply->to_func)){
                // the schedule is an input to function
                templates.push_back("typename TO_FUNC");
                arguments.push_back("TO_FUNC to_func");
            } else {
                // the input is an input to vertexset
                arguments.push_back("VertexSubset<NodeID>* to_vertexset");
            }
        }

        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)){
            auto apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply);

            if (apply_expr->push_to_function_ != ""){
                templates.push_back("typename PUSH_TO_FUNC");
                arguments.push_back("PUSH_TO_FUNC push_to_func");
            }
        }


        templates.push_back("typename APPLY_FUNC");
        arguments.push_back("APPLY_FUNC apply_func");

        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)){
            auto apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply);
            templates.push_back("typename PUSH_APPLY_FUNC");
            arguments.push_back("PUSH_APPLY_FUNC push_apply_func");
        }

        oss_ << "template <";

        bool first = true;
        for (auto  temp : templates){
            if (first){
                oss_ << temp << " ";
                first = false;
            }
            else
                oss_ << ", " << temp;
        }
        oss_ << "> ";
        oss_ << "VertexSubset<NodeID>* " << func_name << "(";

        first = true;
        for (auto arg : arguments) {
            if (first){
                oss_ << arg << " ";
                first = false;
            }
            else
                oss_ << ", " << arg;        }

        oss_ << ") " << endl;


    }

    //generates different function name for different schedules
    std::string EdgesetApplyFunctionDeclGenerator::genFunctionName(mir::EdgeSetApplyExpr::Ptr apply) {
        // A total of 48 schedules for the edgeset apply operator for now
        // Direction first: "push", "pull" or "hybrid_dense"
        // Parallel: "parallel" or "serial"
        // Weighted: "" or "weighted"
        // Deduplicate: "deduplicated" or ""
        // From: "" (no from func specified) or "from_vertexset" or "from_filter_func"
        // To: "" or "to_vertexset" or "to_filter_func"
        // Frontier: "" (no frontier tracking) or "with_frontier"
        // Weighted: "" (unweighted) or "weighted"

        string output_name = "edgeset_apply";

        //check direction
        if (mir::isa<mir::PushEdgeSetApplyExpr>(apply)){
            output_name += "_push";
        } else if (mir::isa<mir::PullEdgeSetApplyExpr>(apply)){
            output_name += "_pull";
        } else if (mir::isa<mir::HybridDenseForwardEdgeSetApplyExpr>(apply)){
            output_name += "_hybrid_denseforward";
        } else if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)){
            output_name += "_hybrid_dense";
        }

        //check parallelism specification
        if (apply->is_parallel){
            output_name += "_parallel";
        } else {
            output_name += "_serial";
        }
        
        if (apply->use_sliding_queue){
            output_name += "_sliding_queue";   
        }

        //check if it is weighted
        if (apply->is_weighted){
            output_name += "_weighted";
        }

        // check for deduplication
        if (apply->enable_deduplication){
            output_name += "_deduplicatied";
        }

        if (apply->from_func != ""){
            if (mir_context_->isFunction(apply->from_func)){
                // the schedule is an input from function
                output_name += "_from_filter_func";
            } else {
                // the input is an input from vertexset
                output_name += "_from_vertexset";
            }
        }

        if (apply->to_func != ""){
            if (mir_context_->isFunction(apply->to_func)){
                // the schedule is an input to function
                output_name += "_to_filter_func";
            } else {
                // the input is an input to vertexset
                output_name += "_to_vertexset";
            }
        }

        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)){
            auto apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply);
            if (apply_expr->push_to_function_ != ""){
                if (mir_context_->isFunction(apply->to_func)){
                    // the schedule is an input to function
                    output_name += "_push_to_filter_func";
                } else {
                    // the input is an input to vertexset
                    output_name += "_push_to_vertexset";
                }
            }
        }

        auto apply_func = mir_context_->getFunction(apply->input_function_name);

        if (apply_func->result.isInitialized()){
            //if frontier tracking is enabled (when apply function returns a boolean value)
            output_name += "_with_frontier";
        }

        return output_name;
    }
}