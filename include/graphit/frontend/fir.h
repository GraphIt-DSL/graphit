//
// Created by Yunming Zhang on 1/24/17.
//

#ifndef GRAPHIT_FIR_H
#define GRAPHIT_FIR_H

#endif //GRAPHIT_FIR_H

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>

#include <graphit/frontend/scanner.h>
#include <graphit/frontend/fir_visitor.h>


namespace graphit {
    namespace fir {

        struct FIRNode;

        template <typename T>
        inline bool isa(std::shared_ptr<FIRNode> ptr) {
            return (bool)std::dynamic_pointer_cast<T>(ptr);
        }

        template <typename T>
        inline const std::shared_ptr<T> to(std::shared_ptr<FIRNode> ptr) {
            std::shared_ptr<T> ret = std::dynamic_pointer_cast<T>(ptr);
            //iassert((bool)ret);
            return ret;
        }

        // Base class for front-end intermediate representation.
        struct FIRNode : public std::enable_shared_from_this<FIRNode> {
            typedef std::shared_ptr<FIRNode> Ptr;

            FIRNode() : lineBegin(0), colBegin(0), lineEnd(0), colEnd(0) {}

            template <typename T = FIRNode> std::shared_ptr<T> clone() {
                return to<T>(cloneNode());
            }

            virtual void accept(FIRVisitor *) = 0;

            virtual unsigned getLineBegin() { return lineBegin; }
            virtual unsigned getColBegin() { return colBegin; }
            virtual unsigned getLineEnd() { return lineEnd; }
            virtual unsigned getColEnd() { return colEnd; }

            void setBeginLoc(const Token &);
            void setEndLoc(const Token &);
            void setLoc(const Token &);

            friend std::ostream &operator<<(std::ostream &, FIRNode &);

        protected:
            template <typename T = FIRNode> std::shared_ptr<T> self() {
                return to<T>(shared_from_this());
            }

            virtual void copy(FIRNode::Ptr);

            virtual FIRNode::Ptr cloneNode() = 0;

        private:
            unsigned lineBegin;
            unsigned colBegin;
            unsigned lineEnd;
            unsigned colEnd;
        };

        struct Program : public FIRNode {
            std::vector<FIRNode::Ptr> elems;
            typedef std::shared_ptr<Program> Ptr;
            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Program>());
            }
        protected:
            virtual void copy(FIRNode::Ptr);
            virtual FIRNode::Ptr cloneNode();
        };


        struct Expr : public FIRNode {
            typedef std::shared_ptr<Expr> Ptr;
            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Expr>());
            }
        protected:
            virtual void copy(FIRNode::Ptr);
            virtual FIRNode::Ptr cloneNode();
        };

        struct Stmt : public FIRNode {
            Expr::Ptr expr;
            typedef std::shared_ptr<Stmt> Ptr;
            virtual void accept(FIRVisitor *visitor) {
                visitor->visit(self<Stmt>());
            }
        protected:
            virtual FIRNode::Ptr cloneNode();
        };

    }
}