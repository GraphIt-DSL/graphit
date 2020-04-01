COMMON_FLAGS = -std=gnu++1y -g -I ../../../src/runtime_lib/
CILK_FLAGS = $(COMMON_FLAGS)
OPENMP_FLAGS = $(COMMON_FLAGS)
TBB_FLAGS = $(COMMON_FLAGS)
GRAPHITC_PY = ../../../build/bin/graphitc.py
GRAPHIT_APP_DIR = ./apps
GRAPHIT_SCHEDULE_DIR= ./schedules

ifdef GCC_PAR
# compiling with gcc CILK and OPENMP
PCC = g++
CILK_FLAGS +=  -fcilkplus -lcilkrts -O3 -DCILK
OPENMP_FLAGS += -DOPENMP -fopenmp -O3 

else ifdef ICPC_PAR
# compiling with Intel icpc compiler
PCC = icpc
CILK_FLAGS += -qopenmp -DCILK -O3
OPENMP_FLAGS += -DOPENMP -qopenmp -O3

else ifdef ICPC_TBB
# compiling with Intel icpc compiler
PCC = icpc
TBB_FLAGS += -qopenmp -DTBB -O3 -ltbb
OPENMP_FLAGS += -DOPENMP -qopenmp -O3

else ifdef TBB 
PCC = g++
TBB_FLAGS += -I /data/scratch/yunming/intel_parallel_studio_cluster/parallel_studio_xe_2017/install/compilers_and_libraries_2017.2.174/linux/tbb/include/ -L /data/scratch/yunming/intel_parallel_studio_cluster/parallel_studio_xe_2017/install/compilers_and_libraries_2017.2.174/linux/tbb/lib/intel64/gcc4.7/ -fopenmp -DTBB -O3 -ltbb 
OPENMP_FLAGS += -DOPENMP -qopenmp -O3

else
# compiling serial programs
PCC = g++
CILK_FLAGS += -O3
OPENMP_FLAGS += -O3

endif

#the cpps files that are faster compiled with cilk 
pr_cilk_cpps = pagerank_pull pagerank_pull_segment 
sssp_cilk_cpps = sssp_hybrid_denseforward 
cc_cilk_cpps = cc_hybrid_dense cc_hybrid_dense_bitvec_segment cc_hybrid_dense_bitvec
bfs_cilk_cpps = bfs_hybrid_dense bfs_hybrid_dense_bitvec 


all: 
	make graphit_files
	make cpps

graphit_files:
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/cc.gt -f ${GRAPHIT_SCHEDULE_DIR}/cc_benchmark_cache.gt -o cpps/cc_hybrid_dense_bitvec_segment.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/cc.gt -f ${GRAPHIT_SCHEDULE_DIR}/cc_hybrid_dense_parallel_cas.gt -o cpps/cc_hybrid_dense.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/cc.gt -f ${GRAPHIT_SCHEDULE_DIR}/cc_hybrid_dense_parallel_bitvector.gt -o cpps/cc_hybrid_dense_bitvec.cpp

	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bfs.gt -f ${GRAPHIT_SCHEDULE_DIR}/bfs_hybrid_dense_parallel_cas.gt -o cpps/bfs_hybrid_dense.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bfs.gt -f ${GRAPHIT_SCHEDULE_DIR}/bfs_hybrid_dense_parallel_cas_bitvector.gt -o cpps/bfs_hybrid_dense_bitvec.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bfs.gt  -f ${GRAPHIT_SCHEDULE_DIR}/bfs_push_sliding_queue_parallel_cas.gt -o cpps/bfs_push_slq.cpp

	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp.gt -f ${GRAPHIT_SCHEDULE_DIR}/sssp_hybrid_denseforward_parallel_cas.gt -o cpps/sssp_hybrid_denseforward.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp.gt -f ${GRAPHIT_SCHEDULE_DIR}/sssp_push_parallel_sliding_queue.gt -o cpps/sssp_push_slq.cpp

	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp_delta_stepping.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_with_merge_argv3.gt -o cpps/sssp_delta_stepping_with_merge.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp_delta_stepping.gt -f ${GRAPHIT_SCHEDULE_DIR}/priority_update_eager_no_merge_argv3.gt -o cpps/sssp_delta_stepping_no_merge.cpp

	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/pagerank.gt -f ${GRAPHIT_SCHEDULE_DIR}/pagerank_benchmark_pull_parallel.gt -o cpps/pagerank_pull.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/pagerank.gt -f ${GRAPHIT_SCHEDULE_DIR}/pagerank_benchmark_cache.gt -o cpps/pagerank_pull_segment.cpp

	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/tc.gt -f ${GRAPHIT_SCHEDULE_DIR}/tc_hiroshi.gt -o cpps/tc_hiroshi.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/tc.gt -f ${GRAPHIT_SCHEDULE_DIR}/tc_multiskip.gt -o cpps/tc_multiskip.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/tc.gt -f ${GRAPHIT_SCHEDULE_DIR}/tc_naive.gt -o cpps/tc_naive.cpp

	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bc.gt -f ${GRAPHIT_SCHEDULE_DIR}/bc_SparsePushDensePull.gt -o cpps/bc_SparsePushDensePull.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bc.gt -f ${GRAPHIT_SCHEDULE_DIR}/bc_SparsePushDensePull_bitvector.gt -o cpps/bc_SparsePushDensePull_bitvector.cpp
cpps: $(bfs_cilk_cpps) $(cc_cilk_cpps) $(sssp_cilk_cpps) $(pr_cilk_cpps) openmp_only_cpps


# this two cpps files can only be compiled with openmp to get parallel performance
openmp_only_cpps:
	$(PCC) $(OPENMP_FLAGS) -o bin/bfs_push_slq cpps/bfs_push_slq.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/sssp_push_slq cpps/sssp_push_slq.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/bc_SparsePushDensePull cpps/bc_SparsePushDensePull.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/bc_SparsePushDensePull_bitvector cpps/bc_SparsePushDensePull_bitvector.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/tc_hiroshi cpps/tc_hiroshi.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/tc_multiskip cpps/tc_multiskip.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/tc_naive cpps/tc_naive.cpp

	$(PCC) $(OPENMP_FLAGS) -o bin/sssp_delta_stepping_with_merge cpps/sssp_delta_stepping_with_merge.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/sssp_delta_stepping_no_merge cpps/sssp_delta_stepping_no_merge.cpp
% : cpps/%.cpp
	$(PCC) -o bin/$@ $< $(CILK_FLAGS) $(TBB_FLAGS)

clean:
	rm cpps/*.cpp
	rm bin/*
