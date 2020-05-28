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

    void EdgesetApplyFunctionDeclGenerator::visit(mir::HybridDenseEdgeSetApplyExpr::Ptr hybrid_dense_apply) {
        genEdgeApplyFunctionDeclaration(hybrid_dense_apply);
    }

    void
    EdgesetApplyFunctionDeclGenerator::visit(mir::HybridDenseForwardEdgeSetApplyExpr::Ptr hybrid_dense_forward_apply) {
        genEdgeApplyFunctionDeclaration(hybrid_dense_forward_apply);
    }

    void EdgesetApplyFunctionDeclGenerator::genEdgeApplyFunctionDeclaration(mir::EdgeSetApplyExpr::Ptr apply) {
        auto func_name = genFunctionName(apply);

        // these schedules are still supported by runtime libraries
        if (func_name == "edgeset_apply_push_parallel_sliding_queue_from_vertexset_with_frontier"
                || func_name == "edgeset_apply_push_parallel_sliding_queue_weighted_deduplicatied_from_vertexset_with_frontier"){
            return;
        }


        genEdgeApplyFunctionSignature(apply);
        oss_ << "{ " << endl; //the end of the function declaration
        genEdgeApplyFunctionDeclBody(apply);
        oss_ << "} //end of edgeset apply function " << endl; //the end of the function declaration

    }

    void EdgesetApplyFunctionDeclGenerator::genEdgeApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply) {
        if (mir::isa<mir::PullEdgeSetApplyExpr>(apply)) {
            genEdgePullApplyFunctionDeclBody(apply);
        }

        if (mir::isa<mir::PushEdgeSetApplyExpr>(apply)) {
            genEdgePushApplyFunctionDeclBody(apply);
        }

        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)) {
            genEdgeHybridDenseApplyFunctionDeclBody(apply);
        }

        if (mir::isa<mir::HybridDenseForwardEdgeSetApplyExpr>(apply)) {
            genEdgeHybridDenseForwardApplyFunctionDeclBody(apply);
        }
    }

    void EdgesetApplyFunctionDeclGenerator::setupFlags(mir::EdgeSetApplyExpr::Ptr apply,
                                                       bool & apply_expr_gen_frontier,
                                                       bool &from_vertexset_specified,
                                                       std::string &dst_type) {

        // set up the flag for checking if a from_vertexset has been specified
        if (apply->from_func)
            if (!mir_context_->isFunction(apply->from_func->function_name->name))
                from_vertexset_specified = true;

        // Check if the apply function has a return value
        auto apply_func = mir_context_->getFunction(apply->input_function->function_name->name);
        dst_type = apply->is_weighted ? "d.v" : "d";

        if (apply_func->result.isInitialized()) {
            // build an empty vertex subset if apply function returns
            apply_expr_gen_frontier = true;
        }
    }

    // Set up the global variables numVertices, numEdges, outdegrees
    void EdgesetApplyFunctionDeclGenerator::setupGlobalVariables(mir::EdgeSetApplyExpr::Ptr apply,
                                                                 bool apply_expr_gen_frontier,
                                                                 bool from_vertexset_specified) {
        oss_ << "    int64_t numVertices = g.num_nodes(), numEdges = g.num_edges();\n";


        if (!mir::isa<mir::PullEdgeSetApplyExpr>(apply)) {
//            if (from_vertexset_specified){
//                printIndent();
//                // for push, we use sparse vertexset
//                oss_ << "    long m = from_vertexset->size();\n";
//            }

            //we need to calculate the outdegrees and m if it is hybrid_dense, hybrid_denseforward or push with output
            if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)
                || mir::isa<mir::HybridDenseForwardEdgeSetApplyExpr>(apply)
                    || (mir::isa<mir::PushEdgeSetApplyExpr>(apply) && apply_expr_gen_frontier)) {
                if (from_vertexset_specified) {
                    oss_ << "    from_vertexset->toSparse();" << std::endl;
                    oss_ << "    long m = from_vertexset->size();\n";

                } else {
                    oss_ << "    long m = numVertices; \n";
                }
                oss_ << "    // used to generate nonzero indices to get degrees\n"
                        "    uintT *degrees = newA(uintT, m);\n"
                        "    // We probably need this when we get something that doesn't have a dense set, not sure\n"
                        "    // We can also write our own, the eixsting one doesn't quite work for bitvectors\n"
                        "    //from_vertexset->toSparse();\n"
                        "    {\n";

                if (from_vertexset_specified){
                    oss_ <<  "        ligra::parallel_for_lambda((long)0, (long)m, [&] (long i) {\n"
                            "            NodeID v = from_vertexset->dense_vertex_set_[i];\n"
                            "            degrees[i] = g.out_degree(v);\n"
                            "         });\n"
                            "    }\n"
                            "    uintT outDegrees = sequence::plusReduce(degrees, m);\n";
                } else {
                    oss_ << "        ligra::parallel_for_lambda((long)0, (long)numVertices, [&] (long i) {\n"
                            "            degrees[i] = g.out_degree(i);\n"
                            "        });\n"
                            "    }\n"
                            "    uintT outDegrees = sequence::plusReduce(degrees, m);\n";
                }
            }
            else if (mir::isa<mir::PushEdgeSetApplyExpr>(apply)){
                //we still need to convert the from_vertexset to sparse, and compute m for SparsePush
                // even when it does not return a frontier
                if (from_vertexset_specified) {
                    oss_ << "    from_vertexset->toSparse();" << std::endl;
                    oss_ << "    long m = from_vertexset->size();\n";

                } else {
                    oss_ << "    long m = numVertices; \n";
                }
            }
        }
    }

    // Print the code for traversing the edges in the push direction and return the new frontier
    // the apply_func_name is used for hybrid schedule, when a special push_apply_func is used
    // usually, the apply_func_name is fixed to "apply_func" (see the default argument)
    void EdgesetApplyFunctionDeclGenerator::printPushEdgeTraversalReturnFrontier(
            mir::EdgeSetApplyExpr::Ptr apply,
            bool from_vertexset_specified,
            bool apply_expr_gen_frontier,
            std::string dst_type,
            std::string apply_func_name) {


        //set up logic fo enabling deduplication with CAS on flags (only if it returns a frontier)
        if (apply->enable_deduplication && apply_expr_gen_frontier) {
            oss_ << "    if (g.get_flags_() == nullptr){\n"
//                    "      g.flags_ = new int[numVertices]();\n"
                    "      g.set_flags_(new int[numVertices]());\n"
                    "      ligra::parallel_for_lambda(0, (int)numVertices, [&] (int i) { g.get_flags_()[i]=0; });\n"
                    "    }\n";
        }

        // If apply function has a return value, then we need to return a temporary vertexsubset
        if (apply_expr_gen_frontier) {
            // build an empty vertex subset if apply function returns
            //set up code for outputing frontier for push based edgeset apply operations
            oss_ << "    VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);\n";
            if (from_vertexset_specified){
                oss_ << "    if (numVertices != from_vertexset->getVerticesRange()) {\n"
                        "        cout << \"edgeMap: Sizes Don't match\" << endl;\n"
                        "        abort();\n"
                        "    }\n";
            }

            oss_ <<
                         "    if (outDegrees == 0) return next_frontier;\n"
                         "    uintT *offsets = degrees;\n"
                         "    long outEdgeCount = sequence::plusScan(offsets, degrees, m);\n"
                         "    uintE *outEdges = newA(uintE, outEdgeCount);\n";
        }


        indent();

        printIndent();


        std::string node_id_type = "NodeID";
        if (apply->is_weighted) node_id_type = "WNode";

        std::string for_type = "for";
        if (apply->is_parallel) {
            // for type changes based on grain sizes
            for_type = "ligra::parallel_for_lambda(";
        }


        if (apply->is_parallel) {
            if (from_vertexset_specified)
                oss_ << for_type << "(long)0, (long)m, [&] (long i) {" << std::endl;
            else
                oss_ << for_type << "(NodeID)0, (NodeID)g.num_nodes(), [&] (NodeID s) {" << std::endl;
        } else {

            if (from_vertexset_specified)
                oss_ << for_type << " (long i=0; i < m; i++) {" << std::endl;
            else
                oss_ << for_type << " (NodeID s=0; s < g.num_nodes(); s++) {" << std::endl;
        }

        indent();

        if (from_vertexset_specified){
            oss_ << "    NodeID s = from_vertexset->dense_vertex_set_[i];\n";
        }

        if (apply_expr_gen_frontier){
            oss_ <<  "    int j = 0;\n";
            if (from_vertexset_specified)
                oss_ << "    uintT offset = offsets[i];\n";
            else
                oss_ << "    uintT offset = offsets[s];\n";
        }


        if (apply->from_func && !from_vertexset_specified) {
            printIndent();
            oss_ << "if (from_func(s)){ " << std::endl;
            indent();
        }

        printIndent();

        oss_ << "for(" << node_id_type << " d : g.out_neigh(s)){" << std::endl;


        // print the checks on filtering on sources s
        if (apply->to_func) {
            indent();
            printIndent();

            oss_ << "if";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply->to_func->function_name->name)) {
                //if the input expression is a function call
                oss_ << " (to_func(" << dst_type << ")";

            } else {
                //the input expression is a vertex subset
                oss_ << " (to_vertexset->bool_map_[s] ";
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
            oss_ << apply_func_name << " ( s , d.v, d.w )";
        } else {
            oss_ << apply_func_name << " ( s , d  )";

        }

        if (!apply_expr_gen_frontier) {
            oss_ << ";" << std::endl;

        } else {


            //need to return a frontier
            if (apply->enable_deduplication && apply_expr_gen_frontier) {
                oss_ << " && CAS(&(g.get_flags_()[" << dst_type << "]), 0, 1) ";
            }

            indent();
            //generate the code for adding destination to "next" frontier
            oss_ << " ) { " << std::endl;
            printIndent();
            oss_ << "outEdges[offset + j] = " << dst_type << "; " << std::endl;
            dedent();
            printIndent();
            oss_ << "} else { outEdges[offset + j] = UINT_E_MAX; }" << std::endl;



//            dedent();
//            printIndent();
//            oss_ << "}" << std::endl;
        }



        // end of from filtering
        if (apply->to_func) {
            dedent();
            printIndent();
            oss_ << "} //end of to func" << std::endl;

            if (apply_expr_gen_frontier){
                printIndent();
                oss_ << " else { outEdges[offset + j] = UINT_E_MAX;  }" << std::endl;
            }

        }

        //increment the index for each source vertex
        if (apply_expr_gen_frontier){
            printIndent();
            oss_ << "j++;" << std::endl;
        }

        //end of for loop on the neighbors
        dedent();
        printIndent();
        oss_ << "} //end of for loop on neighbors" << std::endl;

        if (apply->from_func && !from_vertexset_specified) {
            dedent();
            printIndent();
            oss_ << "} //end of from func " << std::endl;
        }


        dedent();
        printIndent();

        if (apply->is_parallel) {

            if (apply->grain_size != 1024){
                oss_ << "}," << apply->grain_size << ");" << std::endl;

            } else {
                oss_ << "});" << std::endl;
            }

        } else {
            oss_ << "}" << std::endl;
        }

        //return a new vertexset if no subset vertexset is returned
        if (apply_expr_gen_frontier) {
            oss_ << "  uintE *nextIndices = newA(uintE, outEdgeCount);\n"
                    "  long nextM = sequence::filter(outEdges, nextIndices, outEdgeCount, nonMaxF());\n"
                    "  free(outEdges);\n"
                    "  free(degrees);\n"
                    "  next_frontier->num_vertices_ = nextM;\n"
                    "  next_frontier->dense_vertex_set_ = nextIndices;\n";

            //set up logic fo enabling deduplication with CAS on flags (only if it returns a frontier)
            if (apply->enable_deduplication && from_vertexset_specified) {
                //clear up the indices that are set
                oss_ << "  ligra::parallel_for_lambda((int)0, (int)nextM, [&] (int i) {\n"
                        "     g.get_flags_()[nextIndices[i]] = 0;\n"
                        "  });\n";
            }
            oss_ << "  return next_frontier;\n";
        }
    }

    void EdgesetApplyFunctionDeclGenerator::printPullEdgeTraversalInnerNeighborLoop(
            mir::EdgeSetApplyExpr::Ptr apply,
            bool from_vertexset_specified,
            bool apply_expr_gen_frontier,
            std::string dst_type,
            std::string apply_func_name,
            bool cache_aware,
            bool numa_aware) {


        //filtering on destination
        if (apply->to_func) {
            printIndent();
            oss_ << "if (to_func(d)){ " << std::endl;
            indent();
        }

        std::string node_id_type = "NodeID";
        if (apply->is_weighted) node_id_type = "WNode";
        printIndent();

        if (cache_aware || numa_aware) {
            oss_ << "for (int64_t ngh = sg->vertexArray[localId]; ngh < sg->vertexArray[localId+1]; ngh++) {\n";
            printIndent();
            oss_ << "  " << node_id_type << " s = sg->edgeArray[ngh];" << std::endl;
        } else {
            oss_ << "for(" << node_id_type << " s : g.in_neigh(d)){" << std::endl;
        }


        // print the checks on filtering on sources s
        if (apply->from_func) {
            indent();
            printIndent();

            oss_ << "if";

            std::string src_type = apply->is_weighted? "s.v" : "s";

            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply->from_func->function_name->name)) {
                //if the input expression is a function call
                oss_ << " (from_func(" << src_type << ")";

            } else {

                //the input expression is a vertex subset
                if (! apply->use_pull_frontier_bitvector){
                    oss_ << " (from_vertexset->bool_map_[" << src_type <<  "] ";
                } else {
                    oss_ << " (bitmap.get_bit(" << src_type << ")";
                }
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
            oss_ << apply_func_name << " ( s.v , d, s.w " << (numa_aware ? ", socketId" : "") << ")";
        } else {
            oss_ << apply_func_name << " ( s , d " << (numa_aware ? ", socketId" : "") << ")";

        }

        if (!apply_expr_gen_frontier) {
            // no need to generate a frontier
            oss_ << ";" << std::endl;
        } else {
            indent();
            //generate the code for adding destination to "next" frontier
            //TODO: fix later
            oss_ << " ) { " << std::endl;
            printIndent();
            oss_ << "next[d] = 1; " << std::endl;
            // generating code for early break
            if (apply->to_func) {
                printIndent();
                oss_ << "if (!to_func(d)) break; " << std::endl;
            }
            dedent();
            printIndent();
            oss_ << "}" << std::endl;
        }



        // end of from filtering
        if (apply->from_func) {
            dedent();
            printIndent();
            oss_ << "}" << std::endl;
        }

        //end of for loop on the neighbors
        dedent();
        printIndent();
        oss_ << "} //end of loop on in neighbors" << std::endl;

        // end of to filtering (filtering on the destination)
        if (apply->to_func) {
            dedent();
            printIndent();
            oss_ << "} //end of to filtering " << std::endl;
        }
    }

    // Iterate through per-socket local buffers and merge the result into the global buffer
    void EdgesetApplyFunctionDeclGenerator::printNumaMerge(mir::EdgeSetApplyExpr::Ptr apply) {
        oss_ << "}// end of per-socket parallel region\n\n";
        auto edgeset_name = mir::to<mir::VarExpr>(apply->target)->var.getName();
        auto merge_reduce = mir_context_->edgeset_to_label_to_merge_reduce[edgeset_name][apply->scope_label_name];
        oss_ << "  ligra::parallel_for_lambda ((int)0, (int)numVertices, [&] (int n) {\n";
        oss_ << "    for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {\n";
        oss_ << "      " << apply->merge_reduce->field_name << "[n] ";
        switch (apply->merge_reduce->reduce_op) {
        case mir::ReduceStmt::ReductionOp::SUM:
            oss_ << "+= local_" << apply->merge_reduce->field_name  << "[socketId][n];\n";
            break;
        case mir::ReduceStmt::ReductionOp::MIN:
            oss_ << "= min(" << apply->merge_reduce->field_name << "[n], local_"
                 << apply->merge_reduce->field_name  << "[socketId][n]);\n";
            break;
        default:
            // TODO: fill in the missing operators when they are actually used
            abort();
        }
        oss_ << "    }\n  });" << std::endl;
    }

    void EdgesetApplyFunctionDeclGenerator::printNumaScatter(mir::EdgeSetApplyExpr::Ptr apply) {
        oss_ << "ligra::parallel_for_lambda((int)0, (int)numVertices, [&] (int n) {\n";
        oss_ << "    for (int socketId = 0; socketId < omp_get_num_places(); socketId++) {\n";
        oss_ << "      local_" << apply->merge_reduce->field_name  << "[socketId][n] = "
             << apply->merge_reduce->field_name << "[n];\n";
        oss_ << "    }\n  });\n";
    }

    // Print the code for traversing the edges in the push direction and return the new frontier
    void EdgesetApplyFunctionDeclGenerator::printPullEdgeTraversalReturnFrontier(
            mir::EdgeSetApplyExpr::Ptr apply,
            bool from_vertexset_specified,
            bool apply_expr_gen_frontier,
            std::string dst_type,
            std::string apply_func_name) {
        // If apply function has a return value, then we need to return a temporary vertexsubset
        if (apply_expr_gen_frontier) {
            // build an empty vertex subset if apply function returns
            apply_expr_gen_frontier = true;

            //        "  long numVertices = g.num_nodes(), numEdges = g.num_edges();\n"
            //        "  long m = from_vertexset->size();\n"

            oss_ << "  VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);\n"
                    "  bool * next = newA(bool, g.num_nodes());\n"
                    "  ligra::parallel_for_lambda((int)0, (int)numVertices, [&] (int i) { next[i] = 0; });\n";
        }

        indent();


        if (apply->from_func) {
            if (!mir_context_->isFunction(apply->from_func->function_name->name)) {
                printIndent();
                oss_ << "from_vertexset->toDense();" << std::endl;
            }
        }

        //generate a bitvector from the dense vertexset (bool map)
        if (from_vertexset_specified && apply->use_pull_frontier_bitvector){
            oss_ << "  Bitmap bitmap(numVertices);\n"
                    "  bitmap.reset();\n"
                    "  ligra::parallel_for_lambda((int) 0, (int)numVertices, 64, [&] (int i){\n"
                    "     int start = i;\n"
                    "     int end = (((i + 64) < numVertices)? (i+64):numVertices);\n"
                    "     for(int j = start; j < end; j++){\n"
                    "        if (from_vertexset->bool_map_[j])\n"
                    "          bitmap.set_bit(j);\n"
                    "     }\n"
                    "  });" << std::endl;
        }

        printIndent();

        // Setup flag for cache_awareness: use cache optimization if the data modified by this apply is segemented
        bool cache_aware = false;
        auto segment_map = mir_context_->edgeset_to_label_to_num_segment;
        for (auto edge_iter = segment_map.begin(); edge_iter != segment_map.end(); edge_iter++) {
            for (auto label_iter = (*edge_iter).second.begin();
            label_iter != (*edge_iter).second.end();
            label_iter++) {
                if ((*label_iter).first == apply->scope_label_name)
                    cache_aware = true;
            }
        }

        // Setup flag for numa_awareness: use numa optimization if the numa flag is set in the merge_reduce data structure
        bool numa_aware = false;
        for (auto iter : mir_context_->edgeset_to_label_to_merge_reduce) {
            for (auto inner_iter : iter.second) {
                if (mir::to<mir::VarExpr>(apply->target)->var.getName() == iter.first
                    && inner_iter.second->numa_aware)
                    numa_aware = true;
            }
        }

        if (numa_aware) {
            printNumaScatter(apply);
        }

        std::string outer_end = "g.num_nodes()";
        std::string iter = "d";

        if (numa_aware || cache_aware) {
            if (numa_aware) {
                std::string num_segment_str = "g.getNumSegments(\"" + apply->scope_label_name + "\");";
                oss_ << "  int numPlaces = omp_get_num_places();\n";
                oss_ << "    int numSegments = g.getNumSegments(\"" + apply->scope_label_name + "\");\n";
                oss_ << "    int segmentsPerSocket = (numSegments + numPlaces - 1) / numPlaces;\n";
                oss_ << "#pragma omp parallel num_threads(numPlaces) proc_bind(spread)\n{\n";
                oss_ << "    int socketId = omp_get_place_num();\n";
                oss_ << "    for (int i = 0; i < segmentsPerSocket; i++) {\n";
                oss_ << "      int segmentId = socketId + i * numPlaces;\n";
                oss_ << "      if (segmentId >= numSegments) break;\n";
            } else {
                oss_ << "  for (int segmentId = 0; segmentId < g.getNumSegments(\"" << apply->scope_label_name
                     << "\"); segmentId++) {\n";
            }
            oss_ << "      auto sg = g.getSegmentedGraph(std::string(\"" << apply->scope_label_name << "\"), segmentId);\n";
            outer_end = "sg->numVertices";
            iter = "localId";
        }

        //genearte the outer for loop
        if (! apply->use_pull_edge_based_load_balance) {
            std::string for_type = "for";
            if (numa_aware) {
                oss_ << "#pragma omp parallel num_threads(omp_get_place_num_procs(socketId)) proc_bind(close)\n{\n";
                oss_ << "#pragma omp for schedule(dynamic, 1024)\n";
            }

            //printIndent();
            if (apply->is_parallel) {
              oss_ << "ligra::parallel_for_lambda((NodeID)0, (NodeID)" << outer_end << ", [&] (NodeID " << iter << ") {" << std::endl;
            } else {
              oss_ << for_type << " ( NodeID " << iter << "=0; " << iter << " < " << outer_end << "; " << iter << "++) {" << std::endl;
            }
            indent();
            if (cache_aware) {
                printIndent();
                oss_ << "NodeID d = sg->graphId[localId];" << std::endl;
            }
        } else {
            // use edge based load balance
            // recursive load balance scheme

            //set up the edge index (in in edge array) for estimating number of edges
            oss_ << "  if (g.get_offsets_() == nullptr) g.SetUpOffsets(true);\n"
                    "  SGOffset * edge_in_index = g.get_offsets_();\n";

            oss_ << "    std::function<void(int,int,int)> recursive_lambda = \n"
                    "    [" << (apply->to_func ?  "&to_func, " : "")
                 << "&apply_func, &g,  &recursive_lambda, edge_in_index" << (cache_aware ? ", sg" : "");
            // capture bitmap and next frontier if needed
            if (from_vertexset_specified) {
                if(apply->use_pull_frontier_bitvector) oss_ << ", &bitmap ";
                else oss_ << ", &from_vertexset";
            }
            if (apply_expr_gen_frontier) oss_ << ", &next ";
            oss_ <<"  ]\n"
                    "    (NodeID start, NodeID end, int grain_size){\n";
            if (cache_aware)
                oss_ << "         if ((start == end-1) || ((sg->vertexArray[end] - sg->vertexArray[start]) < grain_size)){\n"
                        "  for (NodeID localId = start; localId < end; localId++){\n"
                        "    NodeID d = sg->graphId[localId];\n";
            else
                oss_ << "         if ((start == end-1) || ((edge_in_index[end] - edge_in_index[start]) < grain_size)){\n"
                        "  for (NodeID d = start; d < end; d++){\n";
            indent();

        }

        //print the code for inner loop on in neighbors
        printPullEdgeTraversalInnerNeighborLoop(apply, from_vertexset_specified, apply_expr_gen_frontier,
            dst_type, apply_func_name, cache_aware, numa_aware);


        if (! apply->use_pull_edge_based_load_balance) {
            //end of outer for loop
            dedent();
            printIndent();
            if (apply->is_parallel) {
              oss_ << "}); //end of outer for loop" << std::endl;
            } else {
              oss_ << "} //end of outer for loop" << std::endl;
            }
        } else {
            dedent();
            printIndent();
            oss_ << " } //end of outer for loop" << std::endl;
            oss_ << "        } else { // end of if statement on grain size, recursive case next\n"
                    "                  ligra::parallel_invoke([&] { recursive_lambda(start, start + ((end-start) >> 1), grain_size); },\n"
                    "                                         [&] { recursive_lambda(start + ((end-start)>>1), end, grain_size); });\n"
                    "        } \n"
                    "    }; //end of lambda function\n";
            oss_ << "    recursive_lambda(0, " << (cache_aware ? "sg->" : "") << "numVertices, "  <<  apply->pull_edge_based_load_balance_grain_size << ");\n";
        }

        if (numa_aware) {
          oss_ << "} // end of per-socket parallel_for\n";
        }
        if (cache_aware) {
            oss_ << "    } // end of segment for loop\n";
        }

        if (numa_aware) {
            printNumaMerge(apply);
        }

        //return a new vertexset if no subset vertexset is returned
        if (apply_expr_gen_frontier) {
            oss_ << "  next_frontier->num_vertices_ = sequence::sum(next, numVertices);\n"
                    "  next_frontier->bool_map_ = next;\n"
                    "  next_frontier->is_dense = true;\n"
                    "  return next_frontier;\n";
        }

    }


    // Print the code for traversing the edges in the push direction and return the new frontier
    void EdgesetApplyFunctionDeclGenerator::printHybridDenseEdgeTraversalReturnFrontier(
            mir::EdgeSetApplyExpr::Ptr apply,
            bool from_vertexset_specified,
            bool apply_expr_gen_frontier,
            std::string dst_type) {

        oss_ << "    if (m + outDegrees > numEdges / 20) {\n";
        indent();
        //suppplies the pull based apply function
        printPullEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier, dst_type);
        dedent();
        oss_ << "} else {\n";
        indent();
        //uses a special "push_apply_func", which contains synchronizations for the push direction
        printPushEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier, dst_type,
                                             "push_apply_func");
        dedent();
        oss_ << "} //end of else\n";

    }

    // print code for denseforward direction
    void EdgesetApplyFunctionDeclGenerator::printDenseForwardEdgeTraversalReturnFrontier(
            mir::EdgeSetApplyExpr::Ptr apply, bool from_vertexset_specified, bool apply_expr_gen_frontier,
            std::string dst_type) {

        // If apply function has a return value, then we need to return a temporary vertexsubset
        if (apply_expr_gen_frontier) {
            // build an empty vertex subset if apply function returns
            apply_expr_gen_frontier = true;

            //        "  long numVertices = g.num_nodes(), numEdges = g.num_edges();\n"
            //        "  long m = from_vertexset->size();\n"

            oss_ << "  VertexSubset<NodeID> *next_frontier = new VertexSubset<NodeID>(g.num_nodes(), 0);\n"
                    "  bool * next = newA(bool, g.num_nodes());\n"
                    "  ligra::parallel_for_lambda((int)0, (int)numVertices, [&] (int i) { next[i] = 0; });\n";
        }

        indent();

        if (from_vertexset_specified) {
            printIndent();
            oss_ << "from_vertexset->toDense();" << std::endl;
        }

        printIndent();

        std::string for_type = "for";
        if (apply->is_parallel)
            for_type = "parallel_for";

        std::string node_id_type = "NodeID";
        if (apply->is_weighted) node_id_type = "WNode";

        if (apply->is_parallel) {
            oss_ << "ligra::parallel_for_lambda((NodeID)0, (NodeID)g.num_nodes(), [&] (NodeID s) {" << std::endl;
        } else {
            oss_ << "for ( NodeID s=0; s < g.num_nodes(); s++) {" << std::endl;
        }
        indent();

        // print the checks on filtering on sources s
        if (apply->from_func) {
            indent();
            printIndent();

            oss_ << "if";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply->from_func->function_name->name)) {
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

        oss_ << "for(" << node_id_type << " d : g.out_neigh(s)){" << std::endl;
        indent();
        printIndent();

        // print the checks on filtering on sources s
        if (apply->to_func) {
            indent();
            printIndent();

            oss_ << "if";
            //TODO: move this logic in to MIR at some point
            if (mir_context_->isFunction(apply->to_func->function_name->name)) {
                //if the input expression is a function call
                oss_ << " (to_func(" << dst_type << ")";

            } else {
                //the input expression is a vertex subset
                oss_ << " (to_vertexset->bool_map_[s] ";
            }
            oss_ << ") { " << std::endl;
        }

        if (apply_expr_gen_frontier) {
            oss_ << "if( ";
        }

        // generating the C++ code for the apply function call
        if (apply->is_weighted) {
            oss_ << " apply_func ( s , d.v, d.w )";
        } else {
            oss_ << " apply_func ( s , d  )";

        }

        if (!apply_expr_gen_frontier) {
            oss_ << ";" << std::endl;

        } else {
            indent();
            //generate the code for adding destination to "next" frontier
            //TODO: fix later
            oss_ << " ) { " << std::endl;
            printIndent();
            oss_ << "next[" << dst_type <<  "] = 1; " << std::endl;
            dedent();
            printIndent();
            oss_ << "} //end of generating the next frontier" << std::endl;
        }



        // end of to filtering
        if (apply->to_func) {
            dedent();
            printIndent();
            oss_ << "} // end of if to_func filtering" << std::endl;
        }


        dedent();
        printIndent();
        oss_ << "} // end of inner for loop" << std::endl;

        if (apply->from_func) {
            dedent();
            printIndent();
            oss_ << "} // end of if for from_func or from vertexset" << std::endl;
        }

        dedent();
        printIndent();
        if (apply->is_parallel) {
            oss_ << "}); //end of outer for loop" << std::endl;
        } else {
            oss_ << "} //end of outer for loop" << std::endl;
        }

        //return a new vertexset if no subset vertexset is returned
        if (apply_expr_gen_frontier) {
            oss_ << "  next_frontier->num_vertices_ = sequence::sum(next, numVertices);\n"
                    "  next_frontier->bool_map_ = next;\n"
                    "  return next_frontier;\n";
        }
    }

    void EdgesetApplyFunctionDeclGenerator::printHybridDenseForwardEdgeTraversalReturnFrontier(
            mir::EdgeSetApplyExpr::Ptr apply, bool from_vertexset_specified, bool apply_expr_gen_frontier,
            std::string dst_type) {

        oss_ << "    if (m + outDegrees > numEdges / 20) {\n";
        indent();
        //suppplies the pull based apply function
        printDenseForwardEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier, dst_type);
        dedent();
        oss_ << "} else {\n";
        indent();
        //uses a special "push_apply_func", which contains synchronizations for the push direction
        printPushEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier, dst_type
                                             );
        dedent();
        oss_ << "} //end of else\n";

    }

    void EdgesetApplyFunctionDeclGenerator::genEdgePullApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply) {
        bool apply_expr_gen_frontier = false;
        bool from_vertexset_specified = false;
        string dst_type;
        setupFlags(apply, apply_expr_gen_frontier, from_vertexset_specified, dst_type);
        setupGlobalVariables(apply, apply_expr_gen_frontier, from_vertexset_specified);
        printPullEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier, dst_type);
    }


    // Generate the code for pushed based program
    void EdgesetApplyFunctionDeclGenerator::genEdgePushApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply) {
        bool apply_expr_gen_frontier = false;
        bool from_vertexset_specified = false;
        string dst_type;
        setupFlags(apply, apply_expr_gen_frontier, from_vertexset_specified, dst_type);
        setupGlobalVariables(apply, apply_expr_gen_frontier, from_vertexset_specified);
        printPushEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier, dst_type);
    }

    void EdgesetApplyFunctionDeclGenerator::genEdgeHybridDenseApplyFunctionDeclBody(mir::EdgeSetApplyExpr::Ptr apply) {
        bool apply_expr_gen_frontier = false;
        bool from_vertexset_specified = false;
        string dst_type;
        setupFlags(apply, apply_expr_gen_frontier, from_vertexset_specified, dst_type);
        setupGlobalVariables(apply, apply_expr_gen_frontier, from_vertexset_specified);
        printHybridDenseEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier, dst_type);
    }

    void EdgesetApplyFunctionDeclGenerator::genEdgeHybridDenseForwardApplyFunctionDeclBody
            (mir::EdgeSetApplyExpr::Ptr apply) {
        bool apply_expr_gen_frontier = false;
        bool from_vertexset_specified = false;
        string dst_type;
        setupFlags(apply, apply_expr_gen_frontier, from_vertexset_specified, dst_type);
        setupGlobalVariables(apply, apply_expr_gen_frontier, from_vertexset_specified);
        printHybridDenseForwardEdgeTraversalReturnFrontier(apply, from_vertexset_specified, apply_expr_gen_frontier,
                                                           dst_type);
    }

    void EdgesetApplyFunctionDeclGenerator::genEdgeApplyFunctionSignature(mir::EdgeSetApplyExpr::Ptr apply) {
        auto func_name = genFunctionName(apply);

        auto mir_var = std::dynamic_pointer_cast<mir::VarExpr>(apply->target);
        vector<string> templates = vector<string>();
        vector<string> arguments = vector<string>();

        if (apply->is_weighted) {
            arguments.push_back("WGraph & g");
        } else {
            arguments.push_back("Graph & g");
        }

        if (apply->from_func) {
            if (mir_context_->isFunction(apply->from_func->function_name->name)) {
                // the schedule is an input from function
                templates.push_back("typename FROM_FUNC");
                arguments.push_back("FROM_FUNC from_func");
            } else {
                // the input is an input from vertexset
                arguments.push_back("VertexSubset<NodeID>* from_vertexset");
            }
        }

        if (apply->to_func) {
            if (mir_context_->isFunction(apply->to_func->function_name->name)) {
                // the schedule is an input to function
                templates.push_back("typename TO_FUNC");
                arguments.push_back("TO_FUNC to_func");
            } else {
                // the input is an input to vertexset
                arguments.push_back("VertexSubset<NodeID>* to_vertexset");
            }
        }


        templates.push_back("typename APPLY_FUNC");
        arguments.push_back("APPLY_FUNC apply_func");

        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)) {
            auto apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply);

            if (apply_expr->push_to_function_) {
                templates.push_back("typename PUSH_TO_FUNC");
                arguments.push_back("PUSH_TO_FUNC push_to_func");
            }
        }


        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)) {
            auto apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply);
            templates.push_back("typename PUSH_APPLY_FUNC");
            arguments.push_back("PUSH_APPLY_FUNC push_apply_func");
        }

        oss_ << "template <";

        bool first = true;
        for (auto temp : templates) {
            if (first) {
                oss_ << temp << " ";
                first = false;
            } else
                oss_ << ", " << temp;
        }
        oss_ << "> ";
        oss_ << (mir_context_->getFunction(apply->input_function->function_name->name)->result.isInitialized() ?
                 "VertexSubset<NodeID>* " : "void ")  << func_name << "(";

        first = true;
        for (auto arg : arguments) {
            if (first) {
                oss_ << arg << " ";
                first = false;
            } else
                oss_ << ", " << arg;
        }

        oss_ << ") " << endl;


    }

    //generates different function name for different schedules
    // important for cases where we split the kernel iterations and assign different schedules to different iters
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
        auto original_apply_func_name = apply->input_function->function_name->name;

        mir::FuncDecl::Ptr apply_func = mir_context_->getFunction(apply->input_function->function_name->name);

        //check direction
        if (mir::isa<mir::PushEdgeSetApplyExpr>(apply)) {
            output_name += "_push";
        } else if (mir::isa<mir::PullEdgeSetApplyExpr>(apply)) {
            output_name += "_pull";
        } else if (mir::isa<mir::HybridDenseForwardEdgeSetApplyExpr>(apply)) {
            output_name += "_hybrid_denseforward";
        } else if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)) {
            output_name += "_hybrid_dense";
        }

        //check parallelism specification
        if (apply->is_parallel) {
            output_name += "_parallel";
        } else {
            output_name += "_serial";
        }

        if (apply->use_sliding_queue) {
            output_name += "_sliding_queue";
        }

        //check if it is weighted
        if (apply->is_weighted) {
            output_name += "_weighted";
        }

        // check for deduplication
        if (apply->enable_deduplication && apply_func->result.isInitialized()) {
            output_name += "_deduplicatied";
        }

        if (apply->from_func) {
            if (mir_context_->isFunction(apply->from_func->function_name->name)) {
                // the schedule is an input from function
                output_name += "_from_filter_func";
            } else {
                // the input is an input from vertexset
                output_name += "_from_vertexset";
            }
        }

        if (apply->to_func) {
            if (mir_context_->isFunction(apply->to_func->function_name->name)) {
                // the schedule is an input to function
                output_name += "_to_filter_func";
            } else {
                // the input is an input to vertexset
                output_name += "_to_vertexset";
            }
        }

        if (mir::isa<mir::HybridDenseEdgeSetApplyExpr>(apply)) {
            auto apply_expr = mir::to<mir::HybridDenseEdgeSetApplyExpr>(apply);
            if (apply_expr->push_to_function_) {
                if (mir_context_->isFunction(apply->to_func->function_name->name)) {
                    // the schedule is an input to function
                    output_name += "_push_to_filter_func";
                } else {
                    // the input is an input to vertexset
                    output_name += "_push_to_vertexset";
                }
            }
        }


        if (apply_func->result.isInitialized()) {
            //if frontier tracking is enabled (when apply function returns a boolean value)
            output_name += "_with_frontier";
        }

        if (apply->use_pull_frontier_bitvector){
            output_name += "_pull_frontier_bitvector";
        }

        if (apply->use_pull_edge_based_load_balance){
            output_name += "_pull_edge_based_load_balance";
        }

        return output_name;
    }


}
