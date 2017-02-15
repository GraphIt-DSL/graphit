//
// Created by Yunming Zhang on 1/15/17.
//

#include <graphit/frontend/frontend.h>
#include<graphit/frontend/token.h>
#include <graphit/frontend/parser.h>
#include <graphit/frontend/fir_printer.h>
#include <graphit/midend/mir_context.h>

namespace graphit {



    /// Parses, typechecks and turns a given Simit-formated stream into Simit IR.
    int Frontend::parseStream(std::istream &programStream, FIRContext *context) {

        // Lexical and syntactic analyses.
        TokenStream tokens = Scanner().lex(programStream);
        fir::Program::Ptr program = Parser().parse(tokens);

        //prints out the FIR
        std::cout << "fir: " << std::endl;
        std::cout << *program;
        std::cout << std::endl;

        context->setProgram(program);

        return 0;
    }

    /// Parses, typechecks and turns a given Simit-formated string into Simit IR.
    int Frontend::parseString(const std::string &programString) {
        return 0;
    }

    /// Parses, typechecks and turns a given Simit-formated file into Simit IR.
    int Frontend::parseFile(const std::string &filename) {
        return 0;
    }

}