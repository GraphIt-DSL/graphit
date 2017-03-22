//
// Created by Yunming Zhang on 1/14/17.
//

#ifndef GRAPHIT_SCANNER_H
#define GRAPHIT_SCANNER_H

#include <sstream>
#include <string>
#include <vector>
#include <graphit/frontend/token.h>
#include <graphit/frontend/error.h>

namespace graphit {
        class Scanner {
        public:
            Scanner(std::vector<ParseError> *errors) : errors(errors) {}

            TokenStream lex(std::istream &);

        private:
            enum class ScanState {INITIAL, SLTEST, MLTEST};

        private:
            static Token::Type getTokenType(const std::string);

            void reportError(std::string msg, unsigned line, unsigned col) {
                errors->push_back(ParseError(line, col, line, col, msg));
            }

        private:
            std::vector<ParseError> *errors;
            void printDebugInfo(const std::string & tokenString, TokenStream & tokenStream);
        };

}
#endif //GRAPHIT_SCANNER_H