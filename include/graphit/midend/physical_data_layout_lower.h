//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_LOWERPHYSICALDATALAYOUT_H
#define GRAPHIT_LOWERPHYSICALDATALAYOUT_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>

namespace graphit {

    class PhysicalDataLayoutLower {
    public:

        PhysicalDataLayoutLower( MIRContext* mir_context): mir_context_(mir_context) {};
        PhysicalDataLayoutLower(MIRContext* mir_context, Schedule* schedule)
                : schedule_(schedule), mir_context_(mir_context){};


        void lower();

    private:
        Schedule * schedule_ = nullptr;
        MIRContext * mir_context_ = nullptr;
        void genVariableDecls();

        void genStructDecl(const mir::VarDecl::Ptr var_decl,  const PhysicalDataLayout data_layout);
        void genArrayDecl(const mir::VarDecl::Ptr var_decl);
    };


}

#endif //GRAPHIT_LOWERPHYSICALDATALAYOUT_H
