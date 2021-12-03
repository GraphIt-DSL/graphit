//
// Created by Yunming Zhang on 6/13/17.
//

#ifndef GRAPHIT_HIGH_LEVEL_SCHEDULE_H
#define GRAPHIT_HIGH_LEVEL_SCHEDULE_H


#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include "fir.h"
#include <graphit/frontend/low_level_schedule.h>
#include <graphit/frontend/schedule.h>
#include <map>
#include <regex>

#include <graphit/frontend/gpu_schedule.h>


namespace graphit {
    namespace fir {
        namespace high_level_schedule {

            using namespace std;

            class ProgramScheduleNode
                    : public std::enable_shared_from_this<ProgramScheduleNode> {

            public:

                ProgramScheduleNode(graphit::FIRContext *fir_context)
                        : fir_context_(fir_context) {
                    schedule_ = nullptr;
                    dirCompatibilityMap_ = {
                            {"SparsePush", "push"},
                            {"DensePull", "pull"},
                            {"SparsePush-DensePull", "hybrid_dense"},
                            {"DensePull-SparsePush", "hybrid_dense"},
                            {"DensePush-SparsePush", "hybrid_dense_forward"},
                            {"SparsePush-DensePush", "hybrid_dense_forward"}
                    };

                    parallelCompatibilityMap_ = {
                            {"dynamic-vertex-parallel", "parallel"},
                            {"static-vertex-parallel", "parallel"},
                            {"edge-aware-dynamic-vertex-parallel", "parallel"}
                    };

                }

                ~ ProgramScheduleNode(){
                    if (schedule_ != nullptr)
                        delete(schedule_);
                }
                enum class backend_selection_type {
			CODEGEN_CPU,
                        CODEGEN_GPU,

			CODEGEN_INVALID
                };
                
                backend_selection_type backend_selection = backend_selection_type::CODEGEN_CPU; 

                typedef std::shared_ptr<ProgramScheduleNode> Ptr;


                // High level API for fusing together two fields / system vectors as ArrayOfStructs
                ProgramScheduleNode::Ptr fuseFields(string first_field_name,
                                                    string second_field_name);

                // High level API for fusing together multiple fields / system vectors as ArrayOfStructs
                ProgramScheduleNode::Ptr fuseFields(std::vector<std::string> fields);

                // High level API for splitting one loop into two loops
                ProgramScheduleNode::Ptr splitForLoop(string original_loop_label,
                                                      string split_loop1_label,
                                                      string split_loop2_label,
                                                      int split_loop1_range,
                                                      int split_loop2_range);

                ProgramScheduleNode::Ptr fuseForLoop(string original_loop_label1,
                                                     string original_loop_label2,
                                                     string fused_loop_label);

                // fuse the two apply operators and their functions
                // NOTE: the fused apply would replace the first one
                high_level_schedule::ProgramScheduleNode::Ptr fuseApplyFunctions(string original_apply_label1,
                                                                                 string original_apply_label2,
                                                                                 string fused_apply_name);

                //TODO: add high level APIs for fuseApplyFunctions
                //The APIs are documented here https://docs.google.com/document/d/1y-W8HkQKs3pZr5JX5FiI3pIYt8TPbNIX41V0ThZeVTY/edit?usp=sharing
                //See test/c++/low_level_schedule_test.cpp for examples of implementing these functionalities
                //using low level schedule APIs


                // High lvel API for speicifying direction scheduling options for apply
                // A wrapper around setApply for now.
                // Scheduling Options include DensePush, DensePull, SparsePush, SparsePull, SparsePushDensePull, SparsePushDensePush
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyDirection(std::string apply_label, std::string apply_direction);

                // High level API for specifying which intersection method to use
                // Currently it supports five intersection methods:
                //   1. MultiSkipIntersection
                //   2. HiroshiIntersection
                //   3. NaiveIntersection
                //   4. BinarySearch
                //   5. Combination of MultiSkip and Hiroshi
                // If nothing is provided, it uses naive intersection by default
                high_level_schedule::ProgramScheduleNode::Ptr
                configIntersection(std::string apply_label, std::string intersection_option);

                // High level API for configuring par_for grain_size
                // Currently it supports OPENMP parallel for
                // If nothing is provided, it generates default OPENMP for loop.
                high_level_schedule::ProgramScheduleNode::Ptr
                configParForGrainSize(std::string apply_label, int grain_size=0);

                // High lvel API for speicifying parallelization scheduling options for apply
                // A wrapper around setApply for now.
                // Scheduling Options include VertexParallel, EdgeAwareVertexParallel
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyParallelization(std::string apply_label, std::string apply_schedule, int grain_size=1024, std::string direction = "all");

                // High lvel API for speicifying deduplication scheduling options for apply
                // A wrapper around setApply for now.
                // Scheduling Options include Enable and Disable deduplication
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyDeduplication(std::string apply_label, std::string apply_schedule){
                    return setApply(apply_label, apply_schedule);
                }


                // High lvel API for speicifying Data Structure scheduling options for apply
                // A wrapper around setApply for now.
                // Scheduling Options include bitvector (from_vertexset)
                // Deprecated, to be replaced with configApplyDenseVertexSet
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyDataStructure(std::string apply_label, std::string apply_schedule){
                    return setApply(apply_label, apply_schedule);
                }

                // Configures the physical data layout of vertexset
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyDenseVertexSet(std::string label, std::string config, std::string vertexset = "src-vertexset", std::string direction = "all");



                // High level API for specifying the number of segments to partition the graph into.
                // Used for cache and NUMA optimizations
                // Will soon be DEPRECATED, will be replaced with configApplyNumSSGs.
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyNumSegments(std::string apply_label, int num_segment) {
                    return setApply(apply_label, "num_segment", num_segment);
                }

                // High level API for specifying the number of segments for a particular direction
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyNumSSG(std::string apply_label, std::string config, int num_segment, std::string direction="all");

                // High level API for specifying the number of segments for a particular direction
                // the user can specify a string "argv[x]" to use argv[x] as argument to number of segments at runtime
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyNumSSG(std::string apply_label, std::string config, string num_segment_argv, std::string direction="all");

                // High level API for enabling NUMA optimization
                // Deprecated, to be replaced with configApplyNUMA
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyNumaAware(std::string apply_label) {
                    return setApply(apply_label, "numa_aware");
                }

                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyNUMA(std::string apply_label, std::string config, std::string direction = "all");

                // configures the type of priority update
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyPriorityUpdate(std::string apply_label, std::string config);

                //configures the delta parameter for delta-stepping
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyPriorityUpdateDelta(std::string apply_label, int delta);

                //configures the delta parameter for delta-stepping
                high_level_schedule::ProgramScheduleNode::Ptr
                configApplyPriorityUpdateDelta(std::string apply_label, string delta_argv);

                //configures the parameter for merge threshold for buckets used in eager merge priority queue
                high_level_schedule::ProgramScheduleNode::Ptr
                configBucketMergeThreshold(std::string apply_label, int threshold);

                //configures the parameter for number of materialized buckets used in lazy priority queue
                // number of open buckets need to be a power of 2
                high_level_schedule::ProgramScheduleNode::Ptr
                configNumOpenBuckets(std::string apply_label, int num_open_buckets);


                high_level_schedule::ProgramScheduleNode::Ptr
                configNumOpenBuckets(std::string apply_label, std::string num_open_buckets);

                high_level_schedule::ProgramScheduleNode::Ptr
                configBucketMergeThreshold(std::string apply_label, string threshold);

                // High lvel API for speicifying scheduling options for apply
                // Scheduling Options include push, pull, hybrid, enable_deduplication, disable_deduplication, parallel, serial
                high_level_schedule::ProgramScheduleNode::Ptr
                setApply(std::string apply_label, std::string apply_schedule);


                // High lvel API for speicifying scheduling options for apply
                // Scheduling Options include load balance with edge grain size
                high_level_schedule::ProgramScheduleNode::Ptr
                setApply(std::string apply_label, std::string apply_schedule, int param);

                // High lvel API for speicifying scheduling options for vertexset
                // Scheduling Options include sparse, dense
                high_level_schedule::ProgramScheduleNode::Ptr
                setVertexSet(std::string vertexset_label, std::string vertexset_schedule_str);

                Schedule * getSchedule() {
                    return  schedule_;
                }


		// New GPU Scheduling API
		// We currently need two different functions to apply simple and hybrid schedules
		// TODO: Abstract the simple and hybrid schedules into a single class
		void applyGPUSchedule(std::string label_name, gpu_schedule::SimpleGPUSchedule &s1) {
                	backend_selection = backend_selection_type::CODEGEN_GPU; 

			if (schedule_ == nullptr)
				schedule_ = new Schedule();

			gpu_schedule::SimpleGPUSchedule *s1_copy = new gpu_schedule::SimpleGPUSchedule(s1);
			
			schedule_->apply_gpu_schedules[label_name] = s1_copy;
			
		}
		void applyGPUSchedule(std::string label_name, gpu_schedule::HybridGPUSchedule &s2) {
                	backend_selection = backend_selection_type::CODEGEN_GPU; 

			if (schedule_ == nullptr)
				schedule_ = new Schedule();

			gpu_schedule::HybridGPUSchedule *s2_copy = new gpu_schedule::HybridGPUSchedule(s2);
			*s2_copy = s2;
			
			schedule_->apply_gpu_schedules[label_name] = s2_copy;
		}
		

            private:
                graphit::FIRContext * fir_context_;
                Schedule * schedule_;
                // Maps the new direction to the old directions for backward compatibility for now.
                // For example, "SparsePush" would be mapped to "push"
                // This eventually will be deprecated, just keeping it to keep the unit tests working
                std::map<string, string> dirCompatibilityMap_;
                std::map<string, string> parallelCompatibilityMap_;

                void initGraphIterationSpaceIfNeeded(string label);
                int extractIntegerFromString(string input_string);
                int extractArgvNumFromStringArg(string argv_str);
                ApplySchedule createDefaultSchedule(string apply_label);
            };


        }
    }
}

#endif //GRAPHIT_HIGH_LEVEL_SCHEDULE_H
