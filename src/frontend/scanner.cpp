//
// Created by Yunming Zhang on 1/14/17.
//
#include <graphit/frontend/scanner.h>
#include <iostream>
#include <graphit/frontend/token.h>

namespace graphit {

    //key words in the language
    Token::Type Scanner::getTokenType(const std::string token) {
        if (token == "int") return Token::Type::INT;

        // If string does not correspond to a keyword, assume it is an identifier.
        return Token::Type::IDENT;
    }


    TokenStream Scanner::lex(std::istream &programStream) {
        TokenStream tokens;
        unsigned line = 1;
        unsigned col = 1;
        ScanState state = ScanState::INITIAL;
        //outer loop that goes from token to token
        while (programStream.peek() != EOF) {
            std::string tokenString(1, programStream.get());

            // [_ | [a-zA-Z]]+ [a-zA-Z | 0-9 | _]*
            if (programStream.peek() == '_' || std::isalpha(programStream.peek())) {
                while (programStream.peek() == '_' ||
                       std::isalnum(programStream.peek())) {
                    //a token can have _ or a number as content of the token
                    tokenString += programStream.get();
                }

                Token newToken;
                newToken.type = getTokenType(tokenString);
                newToken.lineBegin = line;
                newToken.colBegin = col;
                newToken.lineEnd = line;
                newToken.colEnd = col + tokenString.length() - 1;
                if (newToken.type == Token::Type::IDENT) {
                    newToken.str = tokenString;
                }
                tokens.addToken(newToken);
                col += tokenString.length();
            }else {
                switch(programStream.peek()) {
                    case '=':
                        programStream.get();
                        //  a == b
                        if (programStream.peek() == '=') {
                            programStream.get();
                            tokens.addToken(Token::Type::EQ, line, col, 2);
                            col += 2;
                        } else { // a = b
                            tokens.addToken(Token::Type::ASSIGN, line, col++);
                        }
                        break;
                    case '+':
                        programStream.get();
                        tokens.addToken(Token::Type::PLUS, line, col++);
                        break;
                    case ';':
                        programStream.get();
                        tokens.addToken(Token::Type::SEMICOL, line, col++);
                        break;
                    default: {
                        Token newToken;
                        newToken.type = Token::Type::INT_LITERAL;
                        newToken.lineBegin = line;
                        newToken.colBegin = col;

                        if (programStream.peek() != '.' &&
                            !std::isdigit(programStream.peek())) {
                            std::stringstream errMsg;
                            std::cout << "unexpected symbol '"
                                      << (char) programStream.peek() << std::endl;

                            while (programStream.peek() != EOF &&
                                   !std::isspace(programStream.peek())) {
                                programStream.get();
                                ++col;
                            }
                            break;
                        }
                    }
                }
            }

        }

        if (state != ScanState::INITIAL) {
            std::cout << "unclosed test" << std::endl;
        }

        tokens.addToken(Token::Type::END, line, col);
        return tokens;
    }
}