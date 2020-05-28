COMMON_FLAGS = -std=c++14 -g -I ../../../src/runtime_lib/ 
CILK_FLAGS = $(COMMON_FLAGS)
OPENMP_FLAGS = $(COMMON_FLAGS)
GRAPHITC_PY = ../../../build/bin/graphitc.py
GRAPHIT_APP_DIR = ../../../apps
GRAPHIT_SCHEDULE_DIR= ../../../test/input_with_schedules

ifdef GCC_PAR
# compiling with gcc CILK and OPENMP
PCC = g++
CILK_FLAGS +=  -fcilkplus -lcilkrts -O2 -DCILK
OPENMP_FLAGS += -DOPENMP -fopenmp -O3 

else ifdef ICPC_PAR
# compiling with Intel icpc compiler
PCC = icpc
CILK_FLAGS += -DCILK -O3
OPENMP_FLAGS += -DOPENMP -qopenmp -O3

else
# compiling serial programs
PCC = g++
CILK_FLAGS += -O3
OPENMP_FLAGS += -O3

endif

#the cpps files that are faster compiled with openmp
sssp_delta_stepping_cpps = sssp_delta_stepping_with_merge sssp_delta_stepping_no_merge
ppsp_delta_stepping_cpps = ppsp_delta_stepping_with_merge ppsp_delta_stepping_no_merge
astar_cpps = astar_with_merge astar_no_merge

all: 
	make graphit_files
	make cpps


#python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/cf.gt -f ${GRAPHIT_SCHEDULE_DIR}/cf_pull_parallel_load_balance_segment_argv.gt -o cpps/cf_pull_load_balance_segment.cpp

k_core_graphit:
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/k_core.gt -f ${GRAPHIT_SCHEDULE_DIR}/k_core_const_sum_reduce.gt -o cpps/k_core_const_sum_reduce.cpp

set_cover_graphit:
	python ${GRAPHITC_PY} -f ${GRAPHIT_APP_DIR}/set_cover.gt  -o cpps/set_cover.cpp

graphit_files:k_core_graphit set_cover_graphit
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp_delta_stepping.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_with_merge_argv3.gt -o cpps/sssp_delta_stepping_with_merge.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp_delta_stepping.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_no_merge_argv3.gt -o cpps/sssp_delta_stepping_no_merge.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/ppsp_delta_stepping.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_with_merge_argv4.gt -o cpps/ppsp_delta_stepping_with_merge.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/ppsp_delta_stepping.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_no_merge_argv4.gt -o cpps/ppsp_delta_stepping_no_merge.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/astar.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_with_merge_argv4.gt -o cpps/astar_with_merge.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/astar.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_no_merge_argv4.gt -o cpps/astar_no_merge.cpp

astar:
	$(PCC) $(OPENMP_FLAGS) -o bin/astar_no_merge cpps/astar_no_merge.cpp ${GRAPHIT_APP_DIR}/astar_distance_loader.cpp 
	$(PCC) $(OPENMP_FLAGS) -o bin/astar_with_merge cpps/astar_with_merge.cpp ${GRAPHIT_APP_DIR}/astar_distance_loader.cpp 


set_cover:
	$(PCC) $(CILK_FLAGS) -o bin/set_cover cpps/set_cover.cpp ${GRAPHIT_APP_DIR}/set_cover_extern.cpp 

cpps: k_core_const_sum_reduce set_cover  ${sssp_delta_stepping_cpps} ${ppsp_delta_stepping_cpps}  astar


#$(PCC) $(OPENMP_FLAGS) -o bin/sssp_push_slq cpps/sssp_push_slq.cpp

# Julienne libraries that rely on cilk
cilk_only_cpps:

% : cpps/%.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/$@ $<

clean:
	-rm -r compile*
	-rm cpps/*.cpp
	-rm -r bin/*

