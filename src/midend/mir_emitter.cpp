//
// Created by Yunming Zhang on 2/9/17.
//

#include <graphit/midend/mir_emitter.h>

namespace graphit {


    void MIREmitter::visit(fir::ConstDecl::Ptr const_decl){

        addVarOrConst(const_decl, true);
    };

    void MIREmitter::visit(fir::VarDecl::Ptr var_decl){
        addVarOrConst(var_decl, false);
        auto mir_var_decl = std::make_shared<mir::VarDecl>();
        mir_var_decl->name = var_decl->name->ident;
        mir_var_decl->initVal = emitExpr(var_decl->initVal);
        mir_var_decl->type = emitType(var_decl->type);


        retStmt = mir_var_decl;
    };

    void MIREmitter::visit(fir::ExprStmt::Ptr expr_stmt) {
        auto mir_expr_stmt = std::make_shared<mir::ExprStmt>();
        mir_expr_stmt->expr = emitExpr(expr_stmt->expr);
        retStmt = mir_expr_stmt;
    }

    void MIREmitter::visit(fir::AssignStmt::Ptr assign_stmt) {
        auto mir_assign_stmt = std::make_shared<mir::AssignStmt>();
        //we only have one expression on the left hand size
        assert(assign_stmt->lhs.size() == 1);
        mir_assign_stmt->lhs = emitExpr(assign_stmt->lhs.front());
        mir_assign_stmt->expr = emitExpr(assign_stmt->expr);
        retStmt = mir_assign_stmt;
    }

    void MIREmitter::visit(fir::PrintStmt::Ptr print_stmt) {
        auto mir_print_stmt = std::make_shared<mir::PrintStmt>();
        //we support printing only one argument first
        assert(print_stmt->args.size() == 1);
        mir_print_stmt->expr = emitExpr(print_stmt->args.front());
        retStmt = mir_print_stmt;
    }

    void MIREmitter::visit(fir::SubExpr::Ptr fir_expr){
        auto mir_expr = std::make_shared<mir::SubExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::VertexSetType::Ptr vertex_set_type) {
        const auto mir_vertex_set_type = std::make_shared<mir::VertexSetType>();
        mir_vertex_set_type->element = std::dynamic_pointer_cast<mir::ElementType>(
                emitType(vertex_set_type->element));
        if (! mir_vertex_set_type){
            std::cout << "Error in Emitting MIR VertexSetType " << std::endl;
            return;
        };
        retType = mir_vertex_set_type;
    }

    void MIREmitter::visit(fir::EdgeSetType::Ptr edge_set_type) {
        const auto mir_edgeset_type = std::make_shared<mir::EdgeSetType>();
        mir_edgeset_type->element = std::dynamic_pointer_cast<mir::ElementType>(
                emitType(edge_set_type->edge_element_type));
        if (! mir_edgeset_type){
            std::cout << "Error in Emitting MIR EdgeSetType " << std::endl;
            return;
        };
        auto mir_vector_element_type_list = new std::vector<mir::ElementType::Ptr>();
        for (auto  vertex_element_type : edge_set_type->vertex_element_type_list){
            auto mir_vertex_element_type = emitType(vertex_element_type);
            mir_vector_element_type_list->push_back(
                    std::dynamic_pointer_cast<mir::ElementType>(mir_vertex_element_type));
        }
        mir_edgeset_type->vertex_element_type_list = mir_vector_element_type_list;

        retType = mir_edgeset_type;
    }

    void MIREmitter::visit(fir::ScalarType::Ptr type){
        auto output = std::make_shared<mir::ScalarType>();

        switch (type->type) {
            case fir::ScalarType::Type::INT:
                output->type = mir::ScalarType::Type::INT;
                retType = output;
                break;
            case fir::ScalarType::Type::FLOAT:
                output->type = mir::ScalarType::Type::FLOAT;
                retType = output;
                break;
//            case ScalarType::Type::BOOL:
//                retType = ir::Boolean;
//                break;
//            case ScalarType::Type::COMPLEX:
//                retType = ir::Complex;
//                break;
//            case ScalarType::Type::STRING:
//                retType = ir::String;
//                break;
            default:
                std::cout << "visit(fir::ScalarType) unrecognized scalar type" << std::endl;
                unreachable;
                break;
        }
    }

    void MIREmitter::visit(fir::NDTensorType::Ptr ND_tensor_type) {
        const auto mir_vector_type = std::make_shared<mir::VectorType>();
        //TODO: fix the code to deal with emitting various different types
        if (ND_tensor_type->element != nullptr){
            mir_vector_type->element_type = std::dynamic_pointer_cast<mir::ElementType> (
                    emitType(ND_tensor_type->element));
            assert(mir_vector_type->element_type != nullptr);
        }


        mir_vector_type->vector_element_type = std::dynamic_pointer_cast<mir::ScalarType>(
                emitType(ND_tensor_type->blockType));
        assert(mir_vector_type->vector_element_type != nullptr);


        retType = mir_vector_type;
    }


    void MIREmitter::visit(fir::StmtBlock::Ptr stmt_block){

        //initialize
        std::vector<mir::Stmt::Ptr>* stmts = new std::vector<mir::Stmt::Ptr>();

        for (auto stmt : stmt_block->stmts) {
            //stmt->accept(this);
            stmts->push_back(emitStmt(stmt));
        }

        auto output_stmt_block = std::make_shared<mir::StmtBlock>();
        if (stmt_block->stmts.size() != 0) {
            //output_stmt_block->stmts = ctx->getStatements();
            output_stmt_block->stmts = stmts;
        }


        retStmt = output_stmt_block;
    }

    void MIREmitter::visit(fir::VertexSetAllocExpr::Ptr expr){
        // Currently we only record a size information in the MIR::VertexSetAllocExpr
        const auto mir_vertexsetalloc_expr = std::make_shared<mir::VertexSetAllocExpr>();
        mir_vertexsetalloc_expr->size_expr = emitExpr(expr->numElements);
        retExpr = mir_vertexsetalloc_expr;
    }

    // use of variables
    void MIREmitter::visit(fir::VarExpr::Ptr var_expr){
        auto mir_var_expr = std::make_shared<mir::VarExpr>();
        mir::Var associated_var = ctx->getSymbol(var_expr->ident);
        mir_var_expr->var = associated_var;
        retExpr = mir_var_expr;
    }

    void MIREmitter::visit(fir::AddExpr::Ptr fir_expr){
        auto mir_expr = std::make_shared<mir::AddExpr>();
        mir_expr->lhs = emitExpr(fir_expr->lhs);
        mir_expr->rhs = emitExpr(fir_expr->rhs);
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::CallExpr::Ptr call_expr){
        auto mir_expr = std::make_shared<mir::Call>();
        mir_expr->name = call_expr->func->ident;
        std::vector<mir::Expr::Ptr> args;
        for (auto & fir_arg : call_expr->args){
            const mir::Expr::Ptr mir_arg = emitExpr(fir_arg);
            args.push_back(mir_arg);
        }
        mir_expr->args = args;
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::ElementTypeDecl::Ptr element_type_decl) {
        const auto mir_element_type = std::make_shared<mir::ElementType>();
        mir_element_type->ident = element_type_decl->name->ident;
        addElementType(mir_element_type);
        retType = mir_element_type;
    }

    void MIREmitter::visit(fir::IdentDecl::Ptr ident_decl){
        auto type = emitType(ident_decl->type);
        //TODO: add type info
        retVar = mir::Var(ident_decl->name->ident, type);
        //TODO: retField in the future ??
    }

    void MIREmitter::visit(fir::FuncDecl::Ptr func_decl){
        auto mir_func_decl = std::make_shared<mir::FuncDecl>();
        ctx->scope();
        std::vector<mir::Var> arguments;

        //processing the arguments to the function declaration
        for (auto arg : func_decl->args){
            const mir::Var arg_var = emitVar(arg);
            arguments.push_back(arg_var);
            ctx->addSymbol(arg_var);
        }
        mir_func_decl->args = arguments;

        //Processing the output of the function declaration
        //we assume there is only one argument for easy C++ code generation
        assert(func_decl->results.size() <= 1);
        if (func_decl->results.size()){
            const mir::Var result_var = emitVar(func_decl->results.front());
            mir_func_decl->result = result_var;
            ctx->addSymbol(result_var);
        }

        //Processing the body of the function declaration
        mir::Stmt::Ptr body;
        if (func_decl->body != nullptr) {
            body = emitStmt(func_decl->body);

            //Aiming for not using the Scope Node
            //body = ir::Scope::make(body);
        }

        //cast the output to StmtBody
        mir_func_decl->body = std::dynamic_pointer_cast<mir::StmtBlock>(body);

        ctx->unscope();

        const auto func_name = func_decl->name->ident;
        mir_func_decl->name = func_name;
        //add the constructed function decl to functions
        ctx->addFunction(mir_func_decl);
    }

    void MIREmitter::visit(fir::IntLiteral::Ptr fir_expr){
        auto mir_expr = std::make_shared<mir::IntLiteral>();
        mir_expr->val = fir_expr->val;
        retExpr = mir_expr;
    };

    void MIREmitter::visit(fir::FloatLiteral::Ptr fir_expr){
        auto mir_expr = std::make_shared<mir::FloatLiteral>();
        mir_expr->val = fir_expr->val;
        retExpr = mir_expr;
    };

    mir::Expr::Ptr MIREmitter::emitExpr(fir::Expr::Ptr ptr){
        auto tmpExpr = retExpr;
        //we should get a null when we don't emit the right type of expr
        //retExpr = std::make_shared<mir::Expr>();

        ptr->accept(this);
        const mir::Expr::Ptr ret = retExpr;

        retExpr = tmpExpr;
        return ret;
    };

    mir::Stmt::Ptr MIREmitter::emitStmt(fir::Stmt::Ptr ptr){
        auto tmpStmt = retStmt;
        retStmt = std::make_shared<mir::Stmt>();

        ptr->accept(this);
        const mir::Stmt::Ptr ret = retStmt;

        retStmt = tmpStmt;
        return ret;
    };

    mir::Type::Ptr MIREmitter::emitType(fir::Type::Ptr ptr) {
        auto tmpType = retType;

        ptr->accept(this);
        auto ret = retType;

        retType = tmpType;
        return ret;
    }


    mir::Var MIREmitter::emitVar(fir::IdentDecl::Ptr ptr) {
        auto tmpVar = retVar;

        ptr->accept(this);
        auto ret = retVar;

        retVar = tmpVar;
        return ret;
    }

    void MIREmitter::visit(fir::ElementType::Ptr element_type) {
        const auto mir_element_type = std::make_shared<mir::ElementType>();
        mir_element_type->ident = element_type->ident;
        retType = mir_element_type;
    }


    void MIREmitter::addVarOrConst(fir::VarDecl::Ptr var_decl, bool is_const){
        //TODO: see if there is a cleaner way to do this, constructor may be???
        //construct a var decl variable
        const auto mir_var_decl = std::make_shared<mir::VarDecl>();
        mir_var_decl->initVal = emitExpr(var_decl->initVal);
        mir_var_decl->name = var_decl->name->ident;
        mir_var_decl->type = emitType(var_decl->type);

        //construct a var variable
        const auto mir_var = mir::Var(mir_var_decl->name, mir_var_decl->type);
        ctx->addSymbol(mir_var);


        if (is_const){

            if (std::dynamic_pointer_cast<mir::VectorType>(mir_var_decl->type) != nullptr){
                mir::VectorType::Ptr type = std::dynamic_pointer_cast<mir::VectorType>(mir_var_decl->type);
                if (type->element_type !=nullptr) {
                    // this is a field / system vector associated with an ElementType

                    if (!ctx->updateElementProperties(type->element_type, mir_var_decl))
                        std::cout << "error in adding constant" << std::endl;
                }
            }
            else if (std::dynamic_pointer_cast<mir::VertexSetType>(mir_var_decl->type) != nullptr){
                mir::VertexSetType::Ptr type = std::dynamic_pointer_cast<mir::VertexSetType>(mir_var_decl->type);
                if (mir_var_decl->initVal != nullptr){
                    ctx->updateElementCount(type->element, mir_var_decl->initVal);
                }
            }
            else if (std::dynamic_pointer_cast<mir::EdgeSetType>(mir_var_decl->type) != nullptr){
                mir::EdgeSetType::Ptr type = std::dynamic_pointer_cast<mir::EdgeSetType>(mir_var_decl->type);
                if (mir_var_decl->initVal != nullptr){
                    if (std::dynamic_pointer_cast<fir::LoadExpr>(mir_var_decl->initVal)){
                        const auto init_val = std::dynamic_pointer_cast<fir::LoadExpr>(mir_var_decl->initVal);
                        const auto mir_name_expr = emitExpr(init_val->file_name);
                        ctx->updateElementInputFilename(type->element, mir_name_expr);
                    }
                }
            }
            else{
                mir_var_decl->modifier = "const";
                ctx->addConstant(mir_var_decl);
            }

        } else {
            //regular var decl
            //TODO:: need to figure out what we do here
        }
    }

    void MIREmitter::addElementType(mir::ElementType::Ptr element_type) {
        ctx->addElementType(element_type);
    }




}