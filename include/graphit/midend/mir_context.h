//
// Created by Yunming Zhang on 2/12/17.
//

#ifndef GRAPHIT_PROGRAM_CONTEXT_H
#define GRAPHIT_PROGRAM_CONTEXT_H

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <utility>

#include <graphit/midend/mir.h>

namespace graphit {


        // Data structure that holds the internal representation of the program
        class MIRContext {

        public:
            MIRContext() {
            }


            ~MIRContext() {
            }

//            void scope() {
//                exprSymtable.scope();
//                statements.push_front(std::vector<ir::Stmt>());
//                builder.setInsertionPoint(&statements.front());
//            }
//
//            void unscope() {
//                exprSymtable.unscope();
//                statements.pop_front();
//                builder.setInsertionPoint(statements.size() > 0
//                                          ? &statements.front() : nullptr);
//            }
//
//            void addSymbol(mir::Var var) {
//                addSymbol(var.getName(), var, Symbol::ReadWrite);
//            }
//
//            bool hasSymbol(const std::string &name) {
//                return exprSymtable.contains(name);
//            }


            //void setProgram(mir::Stmt::Ptr program){this->mir_program = program};
            void addConstant(mir::VarDecl::Ptr var_decl){
                constants.push_back(var_decl);
            }

            std::vector<mir::VarDecl::Ptr> getConstants(){
                return constants;
            }

            void addStatement(mir::Stmt::Ptr stmt) {
                statements.front().push_back(stmt);
            }

            std::vector<mir::Stmt::Ptr> * getStatements() {
                return &statements.front();
            }



        private:
            //mir::Program::Ptr mir_program;
            std::vector<mir::VarDecl::Ptr> constants;
            std::list<std::vector<mir::Stmt::Ptr>> statements;

        };

}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
