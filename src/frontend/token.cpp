//
// Created by Yunming Zhang on 1/22/17.
//

#include <cctype>
#include <string>
#include <cstdlib>
#include <iostream>

#include <graphit/frontend/token.h>


namespace graphit {

    std::string Token::tokenTypeString(Token::Type type) {
        switch (type) {
            case Token::Type::END:
                return "end of file";
            case Token::Type::INT_LITERAL:
                return "an integer literal";
            case Token::Type::FLOAT_LITERAL:
                return "a float literal";
            case Token::Type::IDENT:
                return "an identifier";
            case Token::Type::INT:
                return "'int'";
            case Token::Type::SEMICOL:
                return "';'";
            case Token::Type::ASSIGN:
                return "'='";
            case Token::Type::PLUS:
                return "'+'";
            case Token::Type::EQ:
                return "'=='";
            default:
                return "";
        }
    }

    std::string Token::toString() const {
        switch (type) {
            case Token::Type::INT_LITERAL:
                return "'" + std::to_string(num) + "'";
            case Token::Type::FLOAT_LITERAL:
                return "'" + std::to_string(fnum) + "'";
            case Token::Type::IDENT:
                return "'" + str + "'";
            default:
                return tokenTypeString(type);
        }
    }

    std::ostream &operator <<(std::ostream &out, const Token &token) {
        out << "(" << Token::tokenTypeString(token.type);
        switch (token.type) {
            case Token::Type::INT_LITERAL:
                out << ", " << token.num;
                break;
            case Token::Type::FLOAT_LITERAL:
                out << ", " << token.fnum;
                break;
            case Token::Type::IDENT:
                out << ", " << token.str;
                break;
            default:
                break;
        }
        out << ", " << token.lineBegin << ":" << token.colBegin << "-"
            << token.lineEnd << ":" << token.colEnd << ")";
        return out;
    }

    void TokenStream::addToken(Token::Type type, unsigned line,
                               unsigned col, unsigned len) {
        Token newToken;

        newToken.type = type;
        newToken.lineBegin = line;
        newToken.colBegin = col;
        newToken.lineEnd = line;
        newToken.colEnd = col + len - 1;

        tokens.push_back(newToken);
    }

    bool TokenStream::consume(Token::Type type) {
        if (tokens.front().type == type) {
            tokens.pop_front();
            return true;
        }

        return false;
    }

    Token TokenStream::peek(unsigned k) const {
        if (k == 0) {
            return tokens.front();
        }

        std::list<Token>::const_iterator it = tokens.cbegin();
        for (unsigned i = 0; i < k && it != tokens.cend(); ++i, ++it) {}

        if (it == tokens.cend()) {
            Token endToken = Token();
            endToken.type = Token::Type::END;
            return endToken;
        }

        return *it;
    }

    std::ostream &operator <<(std::ostream &out, const TokenStream &tokens) {
        for (auto it = tokens.tokens.cbegin(); it != tokens.tokens.cend(); ++it) {
            out << *it << std::endl;
        }
        return out;
    }

}

