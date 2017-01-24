//
// Created by Yunming Zhang on 1/15/17.
//

#include <graphit/frontend/parser.h>

namespace graphit {

    fir::Program::Ptr Parser::parse(const TokenStream &tokens) {
        this->tokens = tokens;

        decls = SymbolTable();


        return parseProgram();
    }

    // program: {program_element}
    fir::Program::Ptr Parser::parseProgram() {
        auto program = std::make_shared<fir::Program>();

        while (peek().type != Token::Type::END) {
//            const fir::FIRNode::Ptr element = parseProgramElement();
//            if (element) {
//                program->elems.push_back(element);
//            }
        }

        return program;
    }
}