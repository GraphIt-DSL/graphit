//
// Created by Yunming Zhang on 5/10/17.
//

#ifndef GRAPHIT_SCHEDULE_H
#define GRAPHIT_SCHEDULE_H

#include <string>
#include <map>
#include <vector>
#include <graphit/frontend/gpu_schedule.h>
#include <graphit/frontend/cpu_schedule.h>
#include <graphit/frontend/swarm_schedule.h>
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
            bool numa_aware;
            int merge_threshold;
            int num_open_buckets;
        };

        /**
         * User specified schedule object
         */
        class Schedule {
        public:

          enum class BackendID {
            CPU,
            GPU,
            SWARM
          };



          Schedule(BackendID backendId = BackendID::CPU) {
                physical_data_layouts = new std::map<std::string, FieldVectorPhysicalDataLayout>();
                vertexset_data_layout = std::map<std::string, VertexsetPhysicalLayout>();
                graph_iter_spaces = new std::map<std::string, std::vector<GraphIterationSpace> *>();
                schedule_map = std::map<std::string, fir::abstract_schedule::ScheduleObject::Ptr>(); //label to schedule object
                backend_identifier = backendId ;

            };

            ~Schedule() {
                delete physical_data_layouts;
            }


          // initialize or grab an existing schedule object.
          fir::abstract_schedule::ScheduleObject::Ptr initOrGetScheduleObject(std::string apply_label, std::string direction="all") {
            // if a direction is not specified, then just return the schedule, creating a new one if necessary.
            if (direction == "all") {
              if (schedule_map.find(apply_label) == schedule_map.end()) {
                fir::cpu_schedule::SimpleCPUScheduleObject::Ptr
                    cpu_schedule_object = std::make_shared<fir::cpu_schedule::SimpleCPUScheduleObject>();
                schedule_map[apply_label] = cpu_schedule_object;
              }
              return schedule_map[apply_label];
            } else {
              // if a direction is specified, then assert that the existing schedule's direction is the same, otherwise raise error.
              if (schedule_map.find(apply_label) == schedule_map.end()) {
                fir::cpu_schedule::SimpleCPUScheduleObject::Ptr
                    cpu_schedule_object = std::make_shared<fir::cpu_schedule::SimpleCPUScheduleObject>();
                cpu_schedule_object->configCPUDirection(fir::cpu_schedule::SimpleCPUScheduleObject::translateDirection(direction));
                schedule_map[apply_label] = cpu_schedule_object;
                return schedule_map[apply_label];
              } else {
                fir::abstract_schedule::ScheduleObject::Ptr schedule_object = schedule_map[apply_label];
                if (schedule_object->isComposite()) {
                  return schedule_object;
                }
                auto schedule_direction = schedule_object->self<fir::cpu_schedule::SimpleCPUScheduleObject>()->getCPUDirection();
                if (fir::cpu_schedule::SimpleCPUScheduleObject::translateDirection(direction) == schedule_direction) {
                  return schedule_object;
                } else {
                  assert(false && "Simple schedule direction does not match expected.");
                }
              }
            }
          }

          // convert a simple schedule to a hybrid CPU schedule.
          fir::abstract_schedule::ScheduleObject::Ptr convertSimpleToHybrid(fir::abstract_schedule::ScheduleObject::Ptr simple_schedule, std::string hybrid_direction) {
            assert(!simple_schedule->isComposite());
            auto schedule_direction = simple_schedule->self<fir::cpu_schedule::SimpleCPUScheduleObject>()->getCPUDirection();
            auto schedule_copy = simple_schedule->self<fir::cpu_schedule::SimpleCPUScheduleObject>()->cloneSchedule();
            if (hybrid_direction == "SparsePush-DensePull" || hybrid_direction == "DensePull-SparsePush") {
              if(schedule_direction == fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::SPARSE_PUSH) {
                schedule_copy->configCPUDirection(fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::DENSE_PULL);
              } else if (schedule_direction == fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::DENSE_PULL) {
                schedule_copy->configCPUDirection(fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::SPARSE_PUSH);
              } else {
                // trying to convert an existing schedule without either SparsePush or DensePull
                assert(false && "Hybrid direction not valid for current direction.");
              }
            } else if (hybrid_direction == "DensePush-SparsePush" || hybrid_direction == "SparsePush-DensePush") {
              if (schedule_direction == fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::SPARSE_PUSH) {
                schedule_copy->configCPUDirection(fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::DENSE_PUSH);
              } else if (schedule_direction == fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::DENSE_PUSH) {
                schedule_copy->configCPUDirection(fir::cpu_schedule::SimpleCPUScheduleObject::DirectionType::SPARSE_PUSH);
              } else {
                // trying to convert an existing schedule without either SparsePush or DensePush
                assert(false && "Hybrid direction not valid for current direction.");
              }
            } else {
              assert(false && "direction not recognized.");
            }

            fir::cpu_schedule::HybridCPUScheduleObject::Ptr cpu_hybrid_object = std::make_shared<fir::cpu_schedule::HybridCPUScheduleObject>(simple_schedule, schedule_copy);
            return cpu_hybrid_object;
          }


            //TODO: what does it mean??
            std::map<std::string, FieldVectorPhysicalDataLayout> *physical_data_layouts;

            // this is a vector of graph iteration spaces because we can have up to two graph iteration spaces (for hybrid directions)
            std::map<std::string, std::vector<GraphIterationSpace> *> *graph_iter_spaces;
            std::map<std::string, VertexsetPhysicalLayout> vertexset_data_layout;

            std::map<std::string, fir::abstract_schedule::ScheduleObject::Ptr> schedule_map; //label to schedule object
            BackendID backend_identifier;
        };
    }

#endif //GRAPHIT_SCHEDULE_H
