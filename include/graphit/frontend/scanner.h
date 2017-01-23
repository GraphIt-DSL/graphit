//
// Created by Yunming Zhang on 1/14/17.
//

#ifndef GRAPHIT_SCANNER_H
#define GRAPHIT_SCANNER_H

#include <sstream>
#include <string>
#include <vector>
#include <graphit/frontend/token.h>

#endif //GRAPHIT_SCANNER_H
namespace graphit {
    class Scanner {

    public:
        Scanner() {}
        TokenStream lex(std::istream &programStream);

    private:
        enum class ScanState {INITIAL, SLTEST, MLTEST};
        static Token::Type getTokenType(const std::string);

    };
}