//
// Created by Yunming Zhang on 1/15/17.
//

#include <graphit/frontend/parser.h>

namespace graphit {

// Graphit language grammar is documented here in EBNF. Note that '{}' is used
// here to denote zero or more instances of the enclosing term, while '[]' is
// used to denote zero or one instance of the enclosing term.


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


    fir::Program::Ptr Parser::parse(const TokenStream &tokens) {
        this->tokens = tokens;
        decls = SymbolTable();
        return parseProgram();
    }

    // program: {statement} EOF
    fir::Program::Ptr Parser::parseProgram() {
        auto program = std::make_shared<fir::Program>();

        // making sure the last token is of type END (end of file)
        while (peek().type != Token::Type::END) {
//            std::cout << Token::tokenTypeString(peek().type) << std::endl;
            const fir::FIRNode::Ptr element = parseStmt();
            if (element) {
                program->elems.push_back(element);
            }
            break;
        }

        return program;
    }


    // statement:  expression ';'
    fir::Stmt::Ptr Parser::parseStmt() {
        auto stmt = std::make_shared<fir::Stmt>();
        stmt->expr = parseExpr();
        consume(Token::Type::SEMICOL);
        return stmt;
    }

    // expr: term ExprApost
    fir::Expr::Ptr Parser::parseExpr() {
        Token::Type symbol_type = peek().type;
//        if (symbol_type == Token::Type::PLUS || symbol_type == Token::Type::MINUS){
//            parseFactor();
//        }
        fir::Expr::Ptr expr = parseTerm();
        while (symbol_type == Token::Type::PLUS || symbol_type == Token::Type::MINUS){

            auto temp_expr = std::make_shared<fir::BinaryExpr>(); //init

            if (symbol_type == Token::Type::PLUS){
                consume(Token::Type::PLUS);
                temp_expr = std::make_shared<fir::AddExpr>();

            }else {
                consume(Token::Type::MINUS);
                temp_expr = std::make_shared<fir::MinusExpr>();
            }

            temp_expr->lhs = expr;
            temp_expr->rhs = parseTerm();
            expr = temp_expr;
        }

        return expr;
    }

    fir::Expr::Ptr Parser::parseFactor() {
        Token::Type symbol_type = peek().type;
        switch (symbol_type){
            case Token::Type::IDENT:
                consume(Token::Type::IDENT);
                return std::make_shared<fir::Identifier>();
            case Token::Type::INT_LITERAL:
                consume(Token::Type::INT_LITERAL);
                return std::make_shared<fir::IntLiteral>();
            default:
                util::printDebugInfo("Error in parseFactor");
                return std::make_shared<fir::Expr>();
        }
    }


    fir::Expr::Ptr Parser::parseTerm() {
        auto expr = parseFactor();
        Token::Type symbol_type = peek().type;
        while (symbol_type == Token::Type::PLUS || symbol_type == Token::Type::MINUS){
            auto temp_expr = std::make_shared<fir::BinaryExpr>(); //init

            if (symbol_type == Token::Type::PLUS){
                consume(Token::Type::PLUS);
                temp_expr = std::make_shared<fir::AddExpr>();

            }else {
                consume(Token::Type::MINUS);
                temp_expr = std::make_shared<fir::MinusExpr>();
            }
            temp_expr->lhs = expr;
            temp_expr->rhs = parseFactor();
            expr = temp_expr;
        }
        return expr;
    }

}