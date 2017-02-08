//
// Created by Yunming Zhang on 1/22/17.
//

#ifndef GRAPHIT_TOKEN_H
#define GRAPHIT_TOKEN_H

#endif //GRAPHIT_TOKEN_H

#ifndef TOKEN_H
#define TOKEN_H

#include <list>
#include <sstream>
#include <string>

namespace graphit {

    struct Token {
        enum class Type {
            INT_LITERAL,
            FLOAT_LITERAL,
            INT,
            IDENT,
            SEMICOL,
            EQ,
            PLUS,
            MINUS,
            END,
            ASSIGN
        };

        Type        type;
        union {
            int       num;
            double    fnum;
        };
        std::string str;
        unsigned    lineBegin;
        unsigned    colBegin;
        unsigned    lineEnd;
        unsigned    colEnd;

        static std::string tokenTypeString(Token::Type);

        std::string toString() const;

        friend std::ostream &operator <<(std::ostream &, const Token &);
    };

    struct TokenStream {
        void addToken(Token newToken) { tokens.push_back(newToken); }
        void addToken(Token::Type, unsigned, unsigned, unsigned = 1);

        Token peek(unsigned) const;

        void skip() { tokens.pop_front(); }
        bool consume(Token::Type);

        friend std::ostream &operator <<(std::ostream &, const TokenStream &);

    private:
        std::list<Token> tokens;
    };

}

#endif