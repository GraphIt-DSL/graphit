//
// Created by Yunming Zhang on 1/14/17.
//
#include <graphit/frontend/scanner.h>
#include <iostream>
#include <graphit/frontend/token.h>
#include <graphit/utils/util.h>

namespace graphit {

    //key words in the language
    Token::Type Scanner::getTokenType(const std::string token) {
        if (token == "int") return Token::Type::INT;

        // If string does not correspond to a keyword, assume it is an identifier.
        return Token::Type::IDENT;
    }

    TokenStream Scanner::lex(std::istream &program_stream) {
        TokenStream tokens;
        unsigned line = 1;
        unsigned col = 1;
        ScanState state = ScanState::INITIAL;
        std::string accum_token_string = "";

        //outer loop that goes from token to token
        while (program_stream.peek() != EOF) {
            std::string token_string = "";


            // [_ | [a-zA-Z]]+ [a-zA-Z | 0-9 | _]*
            if (program_stream.peek() == '_' || std::isalpha(program_stream.peek())) {
                token_string = program_stream.get();
                while (program_stream.peek() == '_' ||
                       std::isalnum(program_stream.peek())) {
                    //a token can have _ or a number as content of the token
                    token_string += program_stream.get();
                }

                Token newToken;
                newToken.type = getTokenType(token_string);
                newToken.lineBegin = line;
                newToken.colBegin = col;
                newToken.lineEnd = line;
                newToken.colEnd = col + token_string.length() - 1;
                if (newToken.type == Token::Type::IDENT) {
                    newToken.str = token_string;
                }
                tokens.addToken(newToken);
                col += token_string.length();
            }else {
                token_string += program_stream.peek();
                switch(program_stream.peek()) {
                    case '=':
                        program_stream.get();
                        //  a == b
                        if (program_stream.peek() == '=') {
                            program_stream.get();
                            tokens.addToken(Token::Type::EQ, line, col, 2);
                            col += 2;
                        } else { // a = b
                            tokens.addToken(Token::Type::ASSIGN, line, col++);
                        }
                        break;
                    case ' ':
                        program_stream.get();
                        ++col;
                        break;
                    case '\t':
                        program_stream.get();
                        ++col;
                        break;
                    case '+':
                        program_stream.get();
                        tokens.addToken(Token::Type::PLUS, line, col++);
                        break;
                    case ';':
                        program_stream.get();
                        tokens.addToken(Token::Type::SEMICOL, line, col++);
                        break;
                    default: {

                        Token newToken;
                        newToken.type = Token::Type::INT_LITERAL;
                        newToken.lineBegin = line;
                        newToken.colBegin = col;


                        if (!std::isdigit(program_stream.peek())) {
                            std::stringstream errMsg;
                            std::cout << "unexpected symbol :"
                                      << (char) program_stream.peek() << std::endl;

                            while (program_stream.peek() != EOF &&
                                   !std::isspace(program_stream.peek())) {
                                program_stream.get();
                                ++col;
                            }
                            break;
                        }

                        token_string = "";
                        while (std::isdigit(program_stream.peek())) {
                            token_string += program_stream.get();
                            ++col;
                        }

                        char *end;
                        if (newToken.type == Token::Type::INT_LITERAL) {
                            newToken.num = std::strtol(token_string.c_str(), &end, 0);
                        } else {
                            newToken.fnum = std::strtod(token_string.c_str(), &end);
                        }
                        newToken.lineEnd = line;
                        newToken.colEnd = col - 1;
                        tokens.addToken(newToken);
                        break;
                    }
                }
            }
            accum_token_string += token_string;
        }

        if (state != ScanState::INITIAL) {
            std::cout << "unclosed test" << std::endl;
        }

        tokens.addToken(Token::Type::END, line, col);

        //Debug Statement, should be disabled for actual execution
        this->printDebugInfo(accum_token_string, tokens);
        return tokens;
    }

    void Scanner::printDebugInfo(const std::string & token_string, TokenStream & token_stream){
        util::printDebugInfo(("current token string: "  + token_string));
        std::stringstream ss;
        ss << token_stream;
        util::printDebugInfo((ss.str() + "\n ----- \n"));
    }
}