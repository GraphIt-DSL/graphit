

COMMON_FLAGS = -std=c++11 -I ../../../src/runtime_lib/ 
CILK_FLAGS = $(COMMON_FLAGS)
OPENMP_FLAGS = $(COMMON_FLAGS)

ifdef GCC_PAR
PCC = g++
CILK_FLAGS +=  -fcilkplus -lcilkrts -O2 -DCILK
OPENMP_FLAGS += -DOPENMP -fopenmp -O3 

else ifdef ICPC_PAR

PCC = icpc
CILK_FLAGS += -DCILK -O3
OPENMP_FLAGS += -DOPENMP -qopenmp -O3

else

PCC = g++

endif

pr = pagerank_pull pagerank_pull_segment 
sssp = sssp_hybrid_denseforward 
cc = cc_hybrid_dense cc_hybrid_dense_bitvec_segment
bfs = bfs_hybrid_dense bfs_hybrid_dense_bitvec bfs_push_slq
prd = pagerankdelta_hybrid_dense pagerankdelta_hybrid_dense_bitvec_segment pagerankdelta_sparse_push
cf = cf_pull_load_balance_segment 

all: $(cf) $(prd) $(bfs) $(cc) $(sssp) $(pr)

% : cpps/%.cpp
	$(PCC) $(CILK_FLAGS) -o bin/$@ $<

clean:
	rm bin/*


