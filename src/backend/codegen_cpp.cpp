//
// Created by Yunming Zhang on 2/14/17.
//

#include <graphit/backend/codegen_cpp.h>

namespace graphit {
    int CodeGenCPP::genCPP(MIRContext *mir_context) {
        mir_context_ = mir_context;
        genIncludeStmts();
        genElementData();

        //Processing the constants
        for (auto constant : mir_context->getConstants()){

//            if ((std::dynamic_pointer_cast<mir::VectorType>(constant->type)) !=nullptr){
//                mir::VectorType::Ptr type = std::dynamic_pointer_cast<mir::VectorType>(constant->type);
//                // if the constant decl is a field property of an element (system vector)
//                if (type->element_type != nullptr){
//                    genPropertyArrayImplementation(constant);
//                }
//            } else if (std::dynamic_pointer_cast<mir::VertexSetType>(constant->type)) {
//                // if the constant is a vertex set  decl
//                // currently, no code is generated
//            } else {
//                // regular constant declaration
            constant->accept(this);
//            }
        }
        //Processing the functions
        std::map<std::string, mir::FuncDecl::Ptr>::iterator it;
        auto functions = mir_context->getFunctions();

        for ( it = functions.begin(); it != functions.end(); it++ )
        {
            it->second->accept(this);
        }


        oss << std::endl;
        return 0;
    };

    void CodeGenCPP::genIncludeStmts(){
        oss << "#include <iostream> " << std::endl;
    }

    void CodeGenCPP::visit(mir::ExprStmt::Ptr expr_stmt){
        printIndent();
        expr_stmt->expr->accept(this);
        oss << ";" << std::endl;
    }

    void CodeGenCPP::visit(mir::AssignStmt::Ptr assign_stmt){
        printIndent();
        assign_stmt->lhs->accept(this);
        oss << " = ";
        assign_stmt->expr->accept(this);
        oss << ";" << std::endl;
    }

    void CodeGenCPP::visit(mir::PrintStmt::Ptr print_stmt){
        printIndent();
        oss << "std::cout << ";
        print_stmt->expr->accept(this);
        oss << "<< std::endl;" << std::endl;
    }

    void CodeGenCPP::visit(mir::VarDecl::Ptr var_decl){
        printIndent();
        oss << var_decl->modifier << ' ';
        var_decl->type->accept(this);
        oss << var_decl->name << " " ;
        if (var_decl->initVal != nullptr){
            oss << "= ";
            var_decl->initVal->accept(this);
        }
        oss << ";" << std::endl;
    }



    void CodeGenCPP::visit(mir::FuncDecl::Ptr func_decl){

        //generate the return type
        if (func_decl->result.isInitialized()) {
            func_decl->result.getType()->accept(this);

            //insert an additional var_decl for returning result
            const auto var_decl = std::make_shared<mir::VarDecl>();
            var_decl->name = func_decl->result.getName();
            var_decl->type = func_decl->result.getType();
            if (func_decl->body->stmts == nullptr){
                func_decl->body->stmts = new std::vector<mir::Stmt::Ptr>();
            }
            auto it = func_decl->body->stmts->begin();
            func_decl->body->stmts->insert(it, var_decl);

        } else if (func_decl->name == "main"){
            oss << "int ";
        } else {
            //default to int return type
            oss << "void ";
        }

        //generate the function name and left paren
        oss << func_decl->name << "(";

        bool printDelimiter = false;
        for (auto arg : func_decl->args) {
            if (printDelimiter) {
                oss << ", ";
            }

            arg.getType()->accept(this);
            oss << arg.getName();
            printDelimiter = true;
        }
        oss << ") ";



        //if the function has a body
        if (func_decl->body->stmts) {
            oss << std::endl;
            printBeginIndent();
            indent();

            func_decl->body->accept(this);

            //print a return statemetn if there is a result
            if(func_decl->result.isInitialized()){
                printIndent();
                oss << "return " << func_decl->result.getName() << ";" << std::endl;
            }

            dedent();
            printEndIndent();
        }
        oss << ";";
        oss << std::endl;

    };

    void CodeGenCPP::visit(mir::ScalarType::Ptr scalar_type){
        switch (scalar_type->type) {
            case mir::ScalarType::Type::INT:
                oss << "int ";
                break;
            case mir::ScalarType::Type::FLOAT:
                oss << "float ";
                break;
            case mir::ScalarType::Type::BOOL:
                oss << "bool ";
                break;
            case mir::ScalarType::Type::STRING:
                oss << "string ";
                break;
            default:
                break;
        }
    }

    void CodeGenCPP::visit(mir::Call::Ptr call_expr){
        oss << call_expr->name << "(";

        bool printDelimiter = false;

        for (auto arg : call_expr->args){
            if (printDelimiter) {
                oss << ", ";
            }
            arg->accept(this);
            printDelimiter = true;
        }

        oss << ") ";
    };

    void CodeGenCPP::visit(mir::VarExpr::Ptr expr){
        oss << expr->var.getName();
    };

    void CodeGenCPP::visit(mir::AddExpr::Ptr expr){
        oss << '(';
        expr->lhs->accept(this);
        oss << " + ";
        expr->rhs->accept(this);
        oss << ')';
    };
    void CodeGenCPP::visit(mir::SubExpr::Ptr expr){
        oss << '(';
        expr->lhs->accept(this);
        oss << " - ";
        expr->rhs->accept(this);
        oss << ')';
    };

    void CodeGenCPP::visit(mir::FloatLiteral::Ptr expr){
        oss << "(";
        //oss << "(float) ";
        oss << expr->val;
        oss << ") ";
    };

    void CodeGenCPP::visit(mir::IntLiteral::Ptr expr){
        oss << "(";
        //oss << "(int) ";
        oss << expr->val;
        oss << ") ";
    }

    // Materialize the Element Data
    void CodeGenCPP::genElementData() {
        for (auto const & element_type_entry : mir_context_->properties_map_){
            // for each element type
            for (auto const & var_decl : *element_type_entry.second){
                // for each field / system vector of the element
                // generate just array implementation for now
                genPropertyArrayImplementation(var_decl);
            }
        }

    };

    void CodeGenCPP::genPropertyArrayImplementation(mir::VarDecl::Ptr var_decl) {
        // read the name of the array
        const auto name = var_decl->name;

        // read the type of the array
        mir::VectorType::Ptr vector_type = std::dynamic_pointer_cast<mir::VectorType>(var_decl->type);
        assert(vector_type != nullptr);
        mir::ScalarType::Ptr vector_element_type = vector_type->vector_element_type;
        assert(vector_element_type != nullptr);

        // read the size of the array
        const auto size_expr = mir_context_->getElementCount(vector_type->element_type);
        assert(size_expr != nullptr);
        vector_element_type->accept(this);
        // pointer declaration
        oss << "* ";
        oss << "__restrict ";
        oss << name << " = new ";
        vector_element_type->accept(this);
        oss << "[ ";
        size_expr->accept(this);
        oss << "]; " << std::endl;
    }





}
