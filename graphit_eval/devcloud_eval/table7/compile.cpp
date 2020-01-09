#include <graphit/frontend/high_level_schedule.h>
namespace graphit {
void user_defined_schedule (graphit::fir::high_level_schedule::ProgramScheduleNode::Ptr program) {
    program->configApplyDirection("s1", "SparsePush-DensePull")
    ->configApplyParallelization("s1", "dynamic-vertex-parallel")
    ->configApplyDenseVertexSet("s1","bitvector", "src-vertexset", "DensePull");

    program->configApplyDirection("s2", "SparsePush-DensePull")
    ->configApplyParallelization("s2", "dynamic-vertex-parallel")
    ->configApplyDenseVertexSet("s2","bitvector", "src-vertexset", "DensePull");}
}