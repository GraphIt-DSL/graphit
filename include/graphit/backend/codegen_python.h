//
// Created by Ajay Brahmakshatriya on 5/14/2019
//

#ifndef GRAPHIT_CODEGEN_PY_H
#define GRAPHIT_CODEGEN_PY_H

#include <graphit/midend/mir.h>
#include <graphit/midend/mir_visitor.h>
#include <graphit/midend/mir_context.h>
#include <iostream>
#include <sstream>

namespace graphit {
    class CodeGenPython: mir::MIRVisitor{
    public:
        CodeGenPython(std::ostream &input_oss, MIRContext *mir_context, std::string module_name_, std::string module_path_):
                oss(input_oss), mir_context_(mir_context), module_name(module_name_), module_path(module_path_){
            indentLevel = 0;
        }

        int genPython();

    protected:

        virtual void visit(mir::FuncDecl::Ptr);

    private:

        void indent() { ++indentLevel; }
        void dedent() { --indentLevel; }
        void printIndent() { oss << std::string(2 * indentLevel, ' '); }
        void printBeginIndent() { oss << std::string(2 * indentLevel, ' ') << "{" << std::endl; }
        void printEndIndent() { oss << std::string(2 * indentLevel, ' ') << "}"; }
        std::ostream &oss;
        unsigned      indentLevel;

        MIRContext * mir_context_;
	std::string module_name;
	std::string module_path;
    public:
	void generatePythonImports(void);
    };
}

#endif //GRAPHIT_CODEGEN_PY_H
