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

    }
}
