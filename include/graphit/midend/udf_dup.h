#ifndef GRAPHIT_UDF_DUP_H
#define GRAPHIT_UDF_DUP_H

#include <graphit/midend/mir_context.h>

namespace graphit {

    class UDFReuseFinder: public mir::MIRVisitor {
    public:
        UDFReuseFinder(MIRContext *mir_context) : mir_context_(mir_context){

        }

        void lower();

    virtual void visit(mir::EdgeSetApplyExpr::Ptr);

    private:
        MIRContext *mir_context_;

        // Map to store all the usages of a function
	std::map<std::string, std::vector<mir::EdgeSetApplyExpr::Ptr>> udf_usage_map;

    };
}
#endif //GRAPHIT_UDF_DUP_H
