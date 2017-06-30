//
// Created by Yunming Zhang on 6/29/17.
//

#include <graphit/midend/vector_field_properties_analyzer.h>

namespace graphit {


    struct FieldVectorProperty {
        enum class AccessType {
            LOCAL,
            SHARED
        };

        enum class ReadWriteType {
            READ_ONLY,
            WRITE_ONLY,
            READ_AND_WRITE
        };

        AccessType access_type_;
        ReadWriteType  read_write_type;
    };

    struct applyExprVisitor : public mir::MIRVisitor {

    };

    void VectorFieldPropertiesAnalyzer::analyze() {

    }
}

