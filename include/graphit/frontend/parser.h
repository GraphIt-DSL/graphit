//
// Created by Yunming Zhang on 1/15/17.
//

#ifndef GRAPHIT_PARSER_H
#define GRAPHIT_PARSER_H

#include <graphit/frontend/fir.h>
#include <graphit/frontend/token.h>
#include <graphit/utils/scopedmap.h>

namespace graphit {

    class Parser {
    public:
        Parser(std::vector<ParseError> *errors);

        fir::Program::Ptr parse(const TokenStream &);

    private:
        class SyntaxError : public std::exception {
        };

        enum class IdentType {
            GENERIC_PARAM, RANGE_GENERIC_PARAM,
            TUPLE, FUNCTION, OTHER
        };

        typedef util::ScopedMap<std::string, IdentType> SymbolTable;

    private:
        fir::Program::Ptr parseProgram();

        fir::FIRNode::Ptr parseProgramElement();

        fir::ElementTypeDecl::Ptr parseElementTypeDecl();

        std::vector<fir::FieldDecl::Ptr> parseFieldDeclList();

        fir::FieldDecl::Ptr parseFieldDecl();

        fir::FIRNode::Ptr parseExternFuncOrDecl();

        fir::ExternDecl::Ptr parseExternDecl();

        fir::FuncDecl::Ptr parseExternFuncDecl();

        fir::FuncDecl::Ptr parseFuncDecl();

        std::vector<fir::GenericParam::Ptr> parseGenericParams();

        fir::GenericParam::Ptr parseGenericParam();

        std::vector<fir::Argument::Ptr> parseFunctorArgs();

        fir::FuncExpr::Ptr parseFunctorExpr();

        std::vector<fir::Argument::Ptr> parseArguments();

        fir::Argument::Ptr parseArgumentDecl();

        std::vector<fir::IdentDecl::Ptr> parseResults();

        fir::StmtBlock::Ptr parseStmtBlock();

        fir::Stmt::Ptr parseStmt();

        fir::VarDecl::Ptr parseVarDecl();

        fir::ConstDecl::Ptr parseConstDecl();

        fir::IdentDecl::Ptr parseIdentDecl();

        fir::IdentDecl::Ptr parseTensorDecl();

        fir::WhileStmt::Ptr parseWhileStmt();

        fir::DoWhileStmt::Ptr parseDoWhileStmt();

        fir::IfStmt::Ptr parseIfStmt();

        fir::Stmt::Ptr parseElseClause();

        fir::ForStmt::Ptr parseForStmt();

        fir::ForDomain::Ptr parseForDomain();

        fir::PrintStmt::Ptr parsePrintStmt();

        fir::ExprStmt::Ptr parseDeleteStmt();

        fir::ApplyStmt::Ptr parseApplyStmt();

        fir::ExprStmt::Ptr parseExprOrAssignStmt();

        fir::Expr::Ptr parseExpr();

        fir::MapExpr::Ptr parseMapExpr();

        fir::Expr::Ptr parseOrExpr();

        fir::Expr::Ptr parseAndExpr();

        fir::Expr::Ptr parseXorExpr();

        fir::Expr::Ptr parseEqExpr();

        fir::Expr::Ptr parseTerm();

        fir::Expr::Ptr parseAddExpr();

        fir::Expr::Ptr parseMulExpr();

        fir::Expr::Ptr parseNegExpr();

        fir::Expr::Ptr parseExpExpr();

        fir::Expr::Ptr parseTransposeExpr();

        fir::Expr::Ptr parseTensorReadExpr();

        fir::Expr::Ptr parseFieldReadExpr();

        fir::Expr::Ptr parseSetReadExpr();

        fir::Expr::Ptr parseFactor();

        fir::VarExpr::Ptr parseVarExpr();

        fir::RangeConst::Ptr parseRangeConst();

        fir::CallExpr::Ptr parseCallExpr();

        fir::UnnamedTupleReadExpr::Ptr parseUnnamedTupleReadExpr();

        fir::NamedTupleReadExpr::Ptr parseNamedTupleReadExpr();

        fir::Identifier::Ptr parseIdent();

        std::vector<fir::ReadParam::Ptr> parseReadParams();

        fir::ReadParam::Ptr parseReadParam();

        std::vector<fir::Expr::Ptr> parseExprParams();

        fir::Type::Ptr parseType();

        fir::ElementType::Ptr parseElementType();

        fir::SetType::Ptr parseUnstructuredSetType();

        fir::SetType::Ptr parseGridSetType();

        std::vector<fir::Endpoint::Ptr> parseEndpoints();

        fir::Endpoint::Ptr parseEndpoint();

        fir::TupleElement::Ptr parseTupleElement();

        fir::TupleType::Ptr parseNamedTupleType();

        fir::TupleLength::Ptr parseTupleLength();

        fir::TupleType::Ptr parseUnnamedTupleType();

        fir::TensorType::Ptr parseTensorType();

        fir::NDTensorType::Ptr parseVectorBlockType();

        fir::NDTensorType::Ptr parseMatrixBlockType();

        fir::NDTensorType::Ptr parseTensorBlockType();

        fir::ScalarType::Ptr parseTensorComponentType();

        fir::ScalarType::Ptr parseScalarType();

        std::vector<fir::IndexSet::Ptr> parseIndexSets();

        fir::IndexSet::Ptr parseIndexSet();

        fir::SetIndexSet::Ptr parseSetIndexSet();

        fir::Expr::Ptr parseTensorLiteral();

        fir::DenseTensorLiteral::Ptr parseDenseTensorLiteral();

        fir::DenseTensorLiteral::Ptr parseDenseTensorLiteralInner();

        fir::DenseTensorLiteral::Ptr parseDenseMatrixLiteral();

        fir::DenseTensorLiteral::Ptr parseDenseVectorLiteral();

        fir::IntVectorLiteral::Ptr parseDenseIntVectorLiteral();

        fir::FloatVectorLiteral::Ptr parseDenseFloatVectorLiteral();

        int parseSignedIntLiteral();

        double parseSignedFloatLiteral();

        fir::Test::Ptr parseTest();

        //Graphit Set system
        fir::VertexSetType::Ptr parseVertexSetType();

        fir::ListType::Ptr parseListType();

        fir::Type::Ptr parseEdgeSetType();

        fir::NewExpr::Ptr parseNewExpr();

        fir::LoadExpr::Ptr parseLoadExpr();

        fir::IntersectionExpr::Ptr parseIntersectionExpr();
        
        fir::IntersectNeighborExpr::Ptr parseIntersectNeighborExpr();

        // OG Additions
        fir::PriorityQueueType::Ptr parsePriorityQueueType();


        void reportError(const Token &, std::string);

        Token peek(unsigned k = 0) const { return tokens.peek(k); }

        int findFirstOccurence(Token::Type type) const { return tokens.findFirstOccurence(type); }

        void skipTo(std::vector<Token::Type>);

        Token consume(Token::Type);

        bool tryConsume(Token::Type type) { return tokens.consume(type); }

        bool isIntrinsic(std::string func_name) {
            return std::find(intrinsics_.begin(), intrinsics_.end(), func_name) != intrinsics_.end();
        }

    private:
        SymbolTable decls;
        TokenStream tokens;

        //const std::vector<fir::FuncDecl::Ptr> &intrinsics;
        std::vector<std::string> intrinsics_;
        std::vector<ParseError> *errors;

        //fir::Expr::Ptr parseNewExpr();

        void initIntrinsics();

        fir::BreakStmt::Ptr parseBreakStmt();

        fir::ReduceStmt::Ptr
        parseReduceStmt(Token::Type token_type, fir::ReduceStmt::ReductionOp reduce_op, fir::Expr::Ptr expr);
    };
}


#endif //GRAPHIT_PARSER_H
