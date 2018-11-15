

COMMON_FLAGS = -std=c++11 -g -I ../../../src/runtime_lib/ 
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

#the cpps files that are faster compiled with cilk 
pr_cilk_cpps = pagerank_pull pagerank_pull_segment 
sssp_cilk_cpps = sssp_hybrid_denseforward 
cc_cilk_cpps = cc_hybrid_dense cc_hybrid_dense_bitvec_segment
bfs_cilk_cpps = bfs_hybrid_dense bfs_hybrid_dense_bitvec 
prd_cilk_cpps = pagerankdelta_hybrid_dense pagerankdelta_hybrid_dense_bitvec_segment pagerankdelta_sparse_push

#this can only be compiled with icpc for now, not sure why
cf_cilk_cpps = cf_pull_load_balance_segment 

all: 
	make graphit_files
	make cpps

graphit_files:
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/cf.gt -f ${GRAPHIT_SCHEDULE_DIR}/cf_pull_parallel_load_balance_segment_argv.gt -o cpps/cf_pull_load_balance_segment.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/pagerankdelta.gt -f ${GRAPHIT_SCHEDULE_DIR}/pagerank_delta_hybrid_dense.gt -o cpps/pagerankdelta_hybrid_dense.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/pagerankdelta.gt -f ${GRAPHIT_SCHEDULE_DIR}/pagerank_delta_benchmark_cache.gt -o cpps/pagerankdelta_hybrid_dense_bitvec_segment.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/pagerankdelta.gt -f ${GRAPHIT_SCHEDULE_DIR}/pagerank_delta_sparse_push_parallel.gt -o cpps/pagerankdelta_sparse_push.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/cc.gt -f ${GRAPHIT_SCHEDULE_DIR}/cc_benchmark_cache.gt -o cpps/cc_hybrid_dense_bitvec_segment.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/cc.gt -f ${GRAPHIT_SCHEDULE_DIR}/cc_hybrid_dense_parallel_cas.gt -o cpps/cc_hybrid_dense.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bfs.gt -f ${GRAPHIT_SCHEDULE_DIR}/bfs_hybrid_dense_parallel_cas.gt -o cpps/bfs_hybrid_dense.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bfs.gt -f ${GRAPHIT_SCHEDULE_DIR}/bfs_hybrid_dense_parallel_cas_bitvector.gt -o cpps/bfs_hybrid_dense_bitvec.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/bfs.gt  -f ${GRAPHIT_SCHEDULE_DIR}/bfs_push_sliding_queue_parallel_cas.gt -o cpps/bfs_push_slq.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp.gt -f ${GRAPHIT_SCHEDULE_DIR}/sssp_hybrid_denseforward_parallel_cas.gt -o cpps/sssp_hybrid_denseforward.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/sssp.gt -f ${GRAPHIT_SCHEDULE_DIR}/sssp_push_parallel_sliding_queue.gt -o cpps/sssp_push_slq.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/pagerank.gt -f ${GRAPHIT_SCHEDULE_DIR}/pagerank_benchmark_pull_parallel.gt -o cpps/pagerank_pull.cpp
	python ${GRAPHITC_PY} -a ${GRAPHIT_APP_DIR}/pagerank.gt -f ${GRAPHIT_SCHEDULE_DIR}/pagerank_benchmark_cache.gt -o cpps/pagerank_pull_segment.cpp

cpps:  $(prd_cilk_cpps) $(bfs_cilk_cpps) $(cc_cilk_cpps) $(sssp_cilk_cpps) $(pr_cilk_cpps) openmp_only_cpps


# this two cpps files can only be compiled with openmp to get parallel performance
openmp_only_cpps:
	$(PCC) $(OPENMP_FLAGS) -o bin/bfs_push_slq cpps/bfs_push_slq.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/sssp_push_slq cpps/sssp_push_slq.cpp
% : cpps/%.cpp
	$(PCC) $(CILK_FLAGS) -o bin/$@ $<

clean:
	rm cpps/*.cpp
	rm bin/*
