//
// Created by Yunming Zhang on 1/15/17.
//

#include <graphit/frontend/frontend.h>
#include <graphit/frontend/parser.h>


namespace graphit {


    /// Parses, typechecks and turns a given Simit-formated stream into Simit IR.
    int Frontend::parseStream(std::istream &programStream, FIRContext *context, std::vector<ParseError> *errors) {

        // Lexical and syntactic analyses.
        TokenStream tokens = Scanner(errors).lex(programStream);
        fir::Program::Ptr program = Parser(errors).parse(tokens);

        // Only emit IR if no syntactic or semantic error was found.
        if (!errors->empty()) {
            std::cout << "Error in parsing: " << std::endl;
            for (auto & error : *errors){
                std::cout << error << std::endl;
            }
            std::stable_sort(errors->begin(), errors->end());
            return 1;
        }

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