//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H

#include <string>
#include <map>
#include <vector>

namespace graphit {

    //TODO: move this into the FieldVectorPhysicalDataLayout class definition
    /** An enum describing a type of physical data layout */
    enum class FieldVectorDataLayoutType {
        ARRAY,
        DICT,
        STRUCT
    };


    // tags for Vertex Data Vectors (Field Vectors)
    struct FieldVectorPhysicalDataLayout {
        std::string var_name;
        FieldVectorDataLayoutType data_layout_type;
        // Records the name of the struct for all fused fields
        std::string fused_struct_name;
    };

    struct VertexsetPhysicalLayout {
        enum class DataLayout {
            SPARSE,
            DENSE
        };
        std::string vertexset_label;
        DataLayout data_layout_type;
    };


    struct Tags {
        enum class PR_Tag {
            //dynamic work-stealing based parallelism
                    WorkStealingPar,
            //static partitioned parallelism
                    StaticPar,
            //serial execution
                    Serial
        };
        enum class PT_Tag {
            // partitioning the graph with a fixed number of vertices
                    FixedVertexCount,
            //partitioning the graph with a flexible number of vertices, but similar number of edges
                    EdgeAwareVertexCount
        };
        enum class FT_Tag {
            //Dense Bitvector
                    BitVector,
            // Sparse Array
                    SparseArray,
            //Dense Boolean Array
                    BoolArray
        };


        //default options
        PR_Tag pr_tag = {PR_Tag::Serial};
        PT_Tag pt_tag = {PT_Tag::FixedVertexCount};
        FT_Tag ft_tag = {FT_Tag::BoolArray};

    };

    struct IntersectionSchedule {
        enum class IntersectionType {
            HIROSHI,
            MULTISKIP,
            COMBINED,
            BINARY,
            NAIVE,
        };

    };

    struct GraphIterationSpace {

        enum class Direction {
            Push,
            Pull
        };

        enum class Dimension {
            SSG,
            BSG,
            OuterIter,
            InnerITer
        };

        std::string scope_label_name;
        // default grain size for parallel execution
        int BSG_grain_size = 1024;

        //number of SSGs
        int num_ssg;

        // Maps a dimension string to its tags
        // The dimensions can be
        // 1) SSG, Segmented Subgraph Dimension (created by partitioning based on InnerIter)
        // 2) BSG, Blocked Subgraph Dimension (created by partitioning based on OuterIter)
        // 3) OuterIter, Outer Loop Iterator, can be source or destination vertices
        // 4) InnerIter, Inner Loop Iterator, can be source or destination vertices
        std::map<Dimension, Tags *> tags_dimension_map_;
        Direction direction = Direction::Pull;
        // a convienience field used for identifying a graph iteration space vector
        // for subsequent configuration APIs
        //default direction to DensePull
        std::string scheduling_api_direction = "DensePull";

    private:
        void initialzieTags(Dimension dimension) {
            tags_dimension_map_[dimension] = new Tags();
        }

    public:
        void setPRTag(Dimension dimension, Tags::PR_Tag tag) {
            //construct the tags for that dimension if it didn't have tags before
            if (tags_dimension_map_.find(dimension) == tags_dimension_map_.end()) {
                initialzieTags(dimension);
            }
            tags_dimension_map_[dimension]->pr_tag = tag;
        }

        void setPTTag(Dimension dimension, Tags::PT_Tag tag) {
            if (tags_dimension_map_.find(dimension) == tags_dimension_map_.end()) {
                initialzieTags(dimension);
            }
            tags_dimension_map_[dimension]->pt_tag = tag;
        }

        void setFTTag(Dimension dimension, Tags::FT_Tag tag) {
            if (tags_dimension_map_.find(dimension) == tags_dimension_map_.end()) {
                initialzieTags(dimension);
            }
            tags_dimension_map_[dimension]->ft_tag = tag;
        }
    };



        // Apply Schedule that encodes the optimizations
        // This is slowly deprecated as we move to Graph Iteration Space based implementation
        struct ApplySchedule {
            enum class DirectionType {
                PUSH,
                PULL,
                HYBRID_DENSE,
                HYBRID_DENSE_FORWARD
            };

            enum class ParType {
                Parallel,
                Serial
            };

            enum class FrontierType {
                Sparse,
                Dense
            };

            enum class DeduplicationType {
                Enable,
                Disable
            };

            enum class OtherOpt {
                QUEUE,
                SLIDING_QUEUE
            };

            enum class PullFrontierType {
                BOOL_MAP,
                BITVECTOR
            };

            enum class PullLoadBalance {
                VERTEX_BASED,
                EDGE_BASED
            };

            enum class PriorityUpdateType {
                EAGER_PRIORITY_UPDATE,
                EAGER_PRIORITY_UPDATE_WITH_MERGE,
                CONST_SUM_REDUCTION_BEFORE_UPDATE,
                REDUCTION_BEFORE_UPDATE
            };

            std::string scope_label_name;
            DirectionType direction_type;
            ParType parallel_type;
            //FrontierType frontier_type;
            DeduplicationType deduplication_type;
            OtherOpt opt;
            PullFrontierType pull_frontier_type;
            PullLoadBalance pull_load_balance_type;
            PriorityUpdateType priority_update_type;

            // the grain size for edge based load balance scheme
            // with the default grain size set to 4096
            int pull_load_balance_edge_grain_size;
            int num_segment;
            int delta;
            int grain_size;
            bool numa_aware;
            int merge_threshold;
            int num_open_buckets;
        };

        /**
         * User specified schedule object
         */
        class Schedule {
        public:
            Schedule() {
                physical_data_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
                intersection_schedules = new std::map<std::string, IntersectionSchedule::IntersectionType >();
                apply_schedules = new std::map<std::string, ApplySchedule>();
                vertexset_data_layout = std::map<std::string, VertexsetPhysicalLayout>();
                graph_iter_spaces = new std::map<std::string, std::vector<GraphIterationSpace> *>();


            };

            ~Schedule() {
                delete physical_data_layouts;
                delete apply_schedules;
            }

            //TODO: what does it mean??
            std::map<std::string, FieldVectorPhysicalDataLayout> *physical_data_layouts;
            //will be slowly replaced with graph iteration space
            std::map<std::string, ApplySchedule> *apply_schedules;

            // this is a vector of graph iteration spaces because we can have up to two graph iteration spaces (for hybrid directions)
            std::map<std::string, std::vector<GraphIterationSpace> *> *graph_iter_spaces;
            std::map<std::string, VertexsetPhysicalLayout> vertexset_data_layout;

            std::map<std::string, IntersectionSchedule::IntersectionType> *intersection_schedules;


        };
    }


#endif //GRAPHIT_SCHEDULE_H
