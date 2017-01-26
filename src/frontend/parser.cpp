//
// Created by Yunming Zhang on 1/15/17.
//

#include <graphit/frontend/parser.h>

namespace graphit {

// Graphit language grammar is documented here in EBNF. Note that '{}' is used
// here to denote zero or more instances of the enclosing term, while '[]' is
// used to denote zero or one instance of the enclosing term.


    fir::Program::Ptr Parser::parse(const TokenStream &tokens) {
        this->tokens = tokens;
        decls = SymbolTable();
        return parseProgram();
    }

    // program: {statement}
    fir::Program::Ptr Parser::parseProgram() {
        auto program = std::make_shared<fir::Program>();

        // making sure the last token is of type END (end of file)
        while (peek().type != Token::Type::END) {
            std::cout << Token::tokenTypeString(peek().type) << std::endl;
            const fir::FIRNode::Ptr element = parseStmt();
            if (element) {
                program->elems.push_back(element);
            }
            break;
        }

        return program;
    }


    // statement:  expression, $
    fir::Stmt::Ptr Parser::parseStmt() {
        auto stmt = std::make_shared<fir::Stmt>();
        stmt->expr = parseExpr();
        consume(Token::Type::SEMICOL);
        return stmt;
    }

    // expr: term ExprApost
    fir::Expr::Ptr Parser::parseExpr() {
        switch(peek().type) {
            case Token::Type::IDENT:
                consume(Token::Type::IDENT);
            case Token::Type::INT_LITERAL:
                consume(Token::Type::INT_LITERAL);
        }

    }

    Token Parser::consume(Token::Type type) {
        const Token token = peek();
        //increment the token stream
        if (!tokens.consume(type)) {
//            reportError(token, Token::tokenTypeString(type));
//            throw SyntaxError();
            std::cout << "error, incorrect type" << std::endl;
        }

        return token;
    }

}