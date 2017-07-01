//
// Created by Yunming Zhang on 6/30/17.
//

#ifndef GRAPHIT_FIELDVECTORPROPERTY_H
#define GRAPHIT_FIELDVECTORPROPERTY_H

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
}

#endif //GRAPHIT_FIELDVECTORPROPERTY_H
