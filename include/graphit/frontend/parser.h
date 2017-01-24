//
// Created by Yunming Zhang on 1/15/17.
//

#ifndef GRAPHIT_PARSER_H
#define GRAPHIT_PARSER_H

#include <graphit/frontend/fir.h>
#include <graphit/frontend/token.h>
#include <graphit/utils/scopedmap.h>



#endif //GRAPHIT_PARSER_H
namespace graphit {

    class Parser {
    public:
        Parser(){};
        fir::Program::Ptr parse(const TokenStream &tokens );

        Token peek(unsigned k = 0) const { return tokens.peek(k); }

    private:
        enum class IdentType {GENERIC_PARAM, RANGE_GENERIC_PARAM,
            TUPLE, FUNCTION, OTHER};
        typedef util::ScopedMap<std::string, IdentType> SymbolTable;


        SymbolTable decls;
        TokenStream tokens;

    private:
        fir::Program::Ptr                   parseProgram();

    };
}