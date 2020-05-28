#include <graphit/frontend/high_level_schedule.h>
namespace graphit {
void user_defined_schedule (graphit::fir::high_level_schedule::ProgramScheduleNode::Ptr program) {
        program->configApplyPriorityUpdate("s1", "eager_priority_update");
        program->configApplyPriorityUpdateDelta("s1", "argv[4]");}
}