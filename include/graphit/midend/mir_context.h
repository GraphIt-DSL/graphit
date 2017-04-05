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

#include <graphit/utils/scopedmap.h>
#include <graphit/midend/mir.h>

namespace graphit {


        // Data structure that holds the internal representation of the program
        class MIRContext {

        public:
            MIRContext() {
            }


            ~MIRContext() {
            }

            void scope() {
                //symbol_table.scope();
                statements.push_front(std::vector<mir::Stmt::Ptr>());
                //builder.setInsertionPoint(&statements.front());
            }

            void unscope() {
                //symbol_table.unscope();
                statements.pop_front();
                //builder.setInsertionPoint(statements.size() > 0 ? &statements.front() : nullptr);
            }

            void addFunction(mir::FuncDecl::Ptr f) {
                functions[f->name] = f;
            }

            bool containsFunction(const std::string &name) const {
                return functions.find(name) != functions.end();
            }

            mir::FuncDecl::Ptr getFunction(const std::string &name) {
                assert(containsFunction(name));
                return functions[name];
            }

                void addSymbol(mir::Var var) {
                symbol_table.insert(var.getName(), var);
            }

            const std::map<std::string, mir::FuncDecl::Ptr> &getFunctions() {
                return functions;
            }

            bool hasSymbol(const std::string &name) {
                return symbol_table.contains(name);
            }

            const mir::Var &getSymbol(const std::string &name) {
                return symbol_table.get(name);
            }


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
            std::map<std::string, mir::FuncDecl::Ptr>  functions;
            util::ScopedMap<std::string, mir::Var> symbol_table;

        };

}

#endif //GRAPHIT_PROGRAM_CONTEXT_H
