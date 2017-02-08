//
// Created by Yunming Zhang on 1/24/17.
//

#include <graphit/frontend/fir.h>
#include <graphit/frontend/token.h>

namespace graphit {
    namespace fir {


        void FIRNode::setBeginLoc(const Token &token) {
            lineBegin = token.lineBegin;
            colBegin = token.colBegin;
        }

        void FIRNode::setEndLoc(const Token &token) {
            lineEnd = token.lineEnd;
            colEnd = token.colEnd;
        }

        void FIRNode::setLoc(const Token &token) {
            setBeginLoc(token);
            setEndLoc(token);
        }

//        void FIRNode::copy(FIRNode::Ptr node) {
//            lineBegin = node->lineBegin;
//            colBegin = node->colBegin;
//            lineEnd = node->lineEnd;
//            colEnd = node->colEnd;
//        }
//
//        void Program::copy(FIRNode::Ptr node) {
//            const auto program = to<Program>(node);
//            FIRNode::copy(program);
//            for (const auto &elem : program->elems) {
//                elems.push_back(elem->clone());
//            }
//        }
//
//        FIRNode::Ptr Program::cloneNode() {
//            const auto node = std::make_shared<Program>();
//            node->copy(shared_from_this());
//            return node;
//        }
//
//        FIRNode::Ptr Stmt::cloneNode() {
//            const auto node = std::make_shared<Stmt>();
//            node->copy(shared_from_this());
//            return node;
//        }
//
//        FIRNode::Ptr IntLiteral::cloneNode() {
//            const auto node = std::make_shared<IntLiteral>();
//            node->copy(shared_from_this());
//            return node;
//        }

    }
}
