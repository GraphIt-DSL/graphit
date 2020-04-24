//
// Created by Yunming Zhang on 1/22/17.
//

#ifndef GRAPHIT_TOKEN_H
#define GRAPHIT_TOKEN_H

#include <list>
#include <sstream>
#include <string>

namespace graphit {

    struct Token {
        enum class Type {
            END,
            UNKNOWN,
            INT_LITERAL,
            FLOAT_LITERAL,
            STRING_LITERAL,
            IDENT,
            AND,
            OR,
            NEG,
            INT,
            UINT,
            UINT_64,
            FLOAT,
            BOOL,
            COMPLEX,
            STRING,
            TENSOR,
            MATRIX,
            VECTOR,
            ELEMENT,
            SET,
            GRID,
            OPAQUE,
            VAR,
            CONST,
            EXTERN,
            EXPORT,
            FUNC,
            INOUT,
            APPLY,
            APPLYMODIFIED,
            MAP,
            TO,
            WITH,
            THROUGH,
            REDUCE,
            WHILE,
            DO,
            IF,
            ELIF,
            ELSE,
            FOR,
            IN,
            BLOCKEND,
            RETURN,
            TEST,
            PRINT,
            PRINTLN,
            NEW,
            DELETE,
            INTERSECTION,
            INTERSECT_NEIGH,
            RARROW,
            LP,
            RP,
            LB,
            RB,
            LC,
            RC,
            LA,
            RA,
            COMMA,
            PERIOD,
            COL,
            SEMICOL,
            ASSIGN,
            PLUS,
            MINUS,
            STAR,
            SLASH,
            DOTSTAR,
            DOTSLASH,
            EXP,
            TRANSPOSE,
            BACKSLASH,
            EQ,
            NE,
            LE,
            GE,
            NOT,
            XOR,
            TRUE,
            FALSE,

            DOUBLE,
            VERTEX_SET,
            EDGE_SET,
            WHERE,
            FILTER,
            LOAD,
            FROM,
            BREAK,
            NUMBER_SIGN,
            MODIFIED,
            PLUS_REDUCE,
            MIN_REDUCE,
            MAX_REDUCE,
            DST_FILTER,
            SRC_FILTER,
            ASYNC_MAX_REDUCE,
            ASYNC_MIN_REDUCE,
            LIST,

	    //OG Additions
            PRIORITY_QUEUE,
	    APPLY_UPDATE_PRIORITY,
            APPLY_UPDATE_PRIORITY_EXTERN,
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

        bool contains(Token::Type tokenType) const;

        //Finds the first occurence of the token type from the stream of tokens and return its' relative index.
        int findFirstOccurence(Token::Type tokenType) const;

        void skip() { tokens.pop_front(); }
        bool consume(Token::Type);

        friend std::ostream &operator <<(std::ostream &, const TokenStream &);

    private:
        std::list<Token> tokens;
    };

}

#endif //GRAPHIT_TOKEN_H
