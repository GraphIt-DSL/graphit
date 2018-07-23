

COMMON_FLAGS = -std=c++11 -I ../../../src/runtime_lib/ 
CILK_FLAGS = $(COMMON_FLAGS)
OPENMP_FLAGS = $(COMMON_FLAGS)

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

pr = pagerank_pull pagerank_pull_segment 
sssp = sssp_hybrid_denseforward 
cc = cc_hybrid_dense cc_hybrid_dense_bitvec_segment
bfs = bfs_hybrid_dense bfs_hybrid_dense_bitvec 
prd = pagerankdelta_hybrid_dense pagerankdelta_hybrid_dense_bitvec_segment pagerankdelta_sparse_push
cf = cf_pull_load_balance_segment 

all:  $(prd) $(bfs) $(cc) $(sssp) $(pr) openmp

openmp:
	$(PCC) $(OPENMP_FLAGS) -o bin/bfs_push_slq cpps/bfs_push_slq.cpp
	$(PCC) $(OPENMP_FLAGS) -o bin/sssp_push_slq cpps/sssp_push_slq.cpp
% : cpps/%.cpp
	$(PCC) $(CILK_FLAGS) -o bin/$@ $<

clean:
	rm bin/*


