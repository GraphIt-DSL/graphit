//
// Created by Yunming Zhang on 6/29/17.
//

#ifndef GRAPHIT_VECTOR_FIELD_PROPERTIES_ANALYZER_H
#define GRAPHIT_VECTOR_FIELD_PROPERTIES_ANALYZER_H

#include <graphit/midend/mir_context.h>
#include <graphit/frontend/schedule.h>
#include <graphit/midend/mir_rewriter.h>

namespace graphit {

    /**
     * Analyze the function declarations used in apply operators
     * and figure out the properties of the field vectors used in the func body
     * This analyzer is useful for later insertion of atomics and change tracking code
     * Note, this analysis only makes sense for functions used for apply operators, and not the generarl functions
     * because the analysis on shared and local requires info on the direction
     */
    class VectorFieldPropertiesAnalyzer {
    public:
        VectorFieldPropertiesAnalyzer(MIRContext *mir_context, Schedule *schedule)
        : schedule_(schedule), mir_context_(mir_context) {};

    void analyze();

    private:
        Schedule *schedule_ = nullptr;
        MIRContext *mir_context_ = nullptr;
    };

}

#endif //GRAPHIT_VECTOR_FIELD_PROPERTIES_ANALYZER_H
