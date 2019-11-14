GraphIt Domain Specific Langauge and Compiler [![Build Status](https://travis-ci.org/GraphIt-DSL/graphit.svg?branch=master)](https://travis-ci.org/GraphIt-DSL/graphit)
==========
GraphIt is a high-performance Graph DSL. [Our website](http://graphit-lang.org) has more detailed tutorials and documentations for the language.

Dependencies
===========

To build GraphIt you need to install
[CMake 3.5.0 or greater](http://www.cmake.org/cmake/resources/software.html). This dependency alone will allow you to build GraphIt and generate high-performance C++ implemenations. Currently, we support both Python 2.7 and Python 3 for the end-to-end tests. 

To compile the generated C++ implementations with support for parallleism, you need CILK and OPENMP. One easy way to set up both CILK and OPENMP is to use intel parallel compiler (icpc). The compiler is free for [students](https://software.intel.com/en-us/qualify-for-free-software/student). There are also open source CILK (g++ >= 5.3.0 with support for Cilk Plus), and [OPENMP](https://www.openmp.org/resources/openmp-compilers-tools/) implementations. 

(Optional) To use NUMA optimizations on multi-socket machines, libnuma needs to be installed (on Ubuntu, sudo apt-get install libnuma-dev). We do note, a good number of optimized implementations do not require enabling NUMA optimizations. You can give GraphIt a try even if you do not have libnuma installed.  

(Optional) To use the python bindings for GraphIt, you need to install the following packages - 
 - python3 (version >= 3.5)
 - scipy (can be installed using pip3)
 - pybind11 (can be installed using pip3)

If you are a mac user who recently upgraded to macOS Mojave, and are having issues with unable to find header files "string.h" or "wchar.h" when using cmake, c++ compiler, or the python scripts that uses the c++ compilers, maybe this [post](https://yunmingzhang.wordpress.com/2019/02/13/mojave-upgrade-c-compilation-and-header-files-missing-issue/) will help. As always, let us know if you have any issues with building and using GraphIt. 

 

Build Graphit
===========

To perform an out-of-tree build of Graphit do:

After you have cloned the directory:

```
    cd graphit
    mkdir build
    cd build
    cmake ..
    make
```
Currently, we do require the build directory to be in the root project directory for some unit tests to work. 
To run the C++ test suite do (all tests should pass):

```
    cd build/bin
    ./graphit_test
```

To run the Python end-to-end test suite:

Start from the root directory and change to the build directory

(All tests would pass, but some would generate error messages from the g++ compiler. This is expected.)
The project tests should support both Python 2.x and Python 3.x.

```
    cd build
    python python_tests/test.py
    python python_tests/test_with_schedules.py
```

(Optional) To test the python bindings, the following extra commands can be run from the GraphIt root directory. You do NOT need this for compiling and running stand alone regular GraphIt programs.

```
    cd build
    export PYTHONPATH=.
    python3 python_tests/pybind_test.py
```


When running `test_with_schedules.py`, commands used for compiling GraphIt files, compiling the generated C++ file, and running the compiled binary file are printed. You can reproduce each test and examine the generated C++ files by typing the printed commands in the shell (make sure you are in the build/bin directory). You can also selectively enable a specific test using the TestSuite commands. We provide examples of enabling a subset of Python tests in the comments of the main function in `test_with_schedules.py`. 

Note when running `test.py`, some error message may be printed during the run that are expected. We have expected to fail tests that print certain error messages. Please check the final output. `test_with_schedules.py` might take a few minutes to run. 

Compile GraphIt Programs
===========
GraphIt compiler currently generates a C++ output file from the .gt input GraphIt programs. 
To compile an input GraphIt file with schedules in the same file (assuming the build directory is in the root project directory).

```
    cd build/bin
    python graphitc.py -f ../../test/input_with_schedules/pagerank_benchmark.gt -o test.cpp
```
To compile an input algorithm file and another separate schedule file (some of the test files have hardcoded paths to test inputs, be sure to modify that or change the directory you run the compiled files)

The example below compiles the algorithm file (../../test/input/pagerank.gt), with a separate schedule file (../../test/input_with_schedules/pagerank_pull_parallel.gt)

```
   cd build/bin
   python graphitc.py -a ../../test/input/pagerank_with_filename_arg.gt -f ../../test/input_with_schedules/pagerank_pull_parallel.gt -o test.cpp
```

Compile and Run Generated C++ Programs
===========
To compile a serial version, you can use reguar g++ with support of c++14 standard to compile the generated C++ file (assuming it is named test.cpp).
 
```
    # assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory
    g++ -std=c++14 -I ../../src/runtime_lib/ -O3 test.cpp  -o test
    ./test ../../test/graphs/4.el
```

To compile a parallel version of the c++ program, you will need both CILK and OPENMP. OPENMP is required for programs using NUMA optimized schedule (configApplyNUMA enabled) and static parallel optimizations (static-vertex-parallel option in configApplyParallelization). All other programs can be compiled with CILK. For analyzing large graphs (e.g., twitter, friendster, webgraph) on NUMA machines, numacl -i all improves the parallel performance. For smaller graphs, such as LiveJournal and Road graphs, not using numactl can be faster. 

```
    # assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory

    # compile and run with CILK
      # icpc
      icpc -std=c++14 -I ../../src/runtime_lib/ -DCILK -O3 test.cpp -o test
    
      # g++ (gcc) with cilk support
      g++ -std=c++14 -I ../../src/runtime_lib/ -DCILK -fcilkplus -lcilkrts -O3 test.cpp -o test
    
      # to run the compiled binary on a small test graph, 4.el
      numactl -i all ./test ../../test/graphs/4.el
    
    # compile and run with OPENMP
      # icpc
      icpc -std=c++14 -I ../../src/runtime_lib/ -DOPENMP -qopenmp -O3 test.cpp -o test
    
      # g++ (gcc) with openmp support
      g++ -std=c++14 -I ../../src/runtime_lib/ -DOPENMP -fopenmp -O3 test.cpp -o test
    
      # to run the compiled binary on a small test graph, 4.el
      numactl -i all ./test ../../test/graphs/4.el
    
    # compile and run with NUMA optimizations (only works with OPENMP and needs libnuma). 
      # Sometimes -lnuma will have to come after the test.cpp file
      # icpc
      icpc -std=c++14 -I ../../src/runtime_lib/ -DOPENMP -DNUMA -qopenmp  -O3 test.cpp -lnuma -o test
    
      # g++ (gcc)
      g++ -std=c++14 -I ../../src/runtime_lib/ -DOPENMP -DNUMA -fopenmp -O3 test.cpp -lnuma -o test
      
      # to run with NUMA enabled on a small test graph, 4.el
      OMP_PLACES=sockets ./test ../../test/graphs/4.el
```

You should see some running times printed. The pagerank example files require a commandline argument for the input graph file. If you see a segfault, then it probably means you did not specify an input graph. 


Evaluate GraphIt's Performance
===========

The algorithms we used for benchmarking, such as PageRank, PageRankDelta, BFS, Connected Components, Single Source Shortest Paths and Collaborative Filtering are in the **apps** directory.
These files include ONLY the algorithm and NO schedules. You need to use the appropriate schedules for the specific algorithm and input graph to get the best performance. 

Detailed instructions for replicating the [*OOPSLA 2018 GraphIt paper*](https://dl.acm.org/citation.cfm?id=3276491) performance is [here](https://github.com/GraphIt-DSL/graphit/blob/master/graphit_eval/GraphIt_Evaluation_Guide.md).
In the OOPSLA paper (Table 8), we described the schedules used for each algorithm on each graph on a dual socket system with Intel Xeon E5-2695 v3 CPUs with 12 cores
each for a total of 24 cores and 48 hyper-threads. The system has 128GB of DDR3-1600 memory
and 30 MB last level cache on each socket, and runs with Transparent Huge Pages (THP) enabled. The best schedule for a different machine can be different. You might need to try a few different set of schedules for the best performance. 

Detailed instructions for replicating our *CGO 2020* PriorityGraph paper is in the **graphit/graphit_eval/priority_graph_cgo2020_eval** directory. 

In the schedules shown in Table 8 of the OOPSLA paper, the keyword ’Program’ and the continuation symbol ’->’ are omitted. ’ca’ is the abbreviation for ’configApply’. Note that configApplyNumSSG uses an integer parameter (X) which is dependent on the graph size and the cache size of a system. For example, the complete schedule used for CC on Twitter graph is the following (X is tuned to the cache size)

```
schedule:
    program->configApplyDirection("s1", "SparsePush-DensePull")->configApplyParallelization("s1", "dynamic-vertex-parallel")->configApplyDenseVertexSet("s1","bitvector", "src-vertexset", "DensePull");
    program->configApplyNumSSG("s1", "fixed-vertex-count",  X, "DensePull");
    
```

The **test/input** and **test/input\_with\_schedules** directories contain many examples of the algorithm and schedule files. Use them as references when writing your own schedule.

 We provide **more detailed instructions on evaluating the code generation and performance capability** of GraphIt in **graphit/graphit_eval/GraphIt_Evaluation_Guide.md**. In the guide, we provide instructions for using a series of scripts that make it easeir for people to evaluate GraphIt.. 

Input Graph Formats
===========

GraphIt reuses [GAPBS input formats](https://github.com/sbeamer/gapbs). Specifically, we have tested with edge list file (.el), weighted edge list file (.wel), binary edge list (.sg), and weighted binary edge list (.wsg) formats. Users can use the converters in GAPBS (GAPBS/src/converter.cc) to convert other graph formats into the supported formats, or convert weighted and unweighted edge list files into their respective binary formats. We have provided sample input graph files in the `graphit/test/graphs/` directory. The python tests use the sample input files. 

Autotuning GraphIt Schedules
===========
Pleaes refer to **README.md** in **graphit/auotune** for more details. 
The auotuner is still somehwat experimental. Please read the [instructions](https://github.com/GraphIt-DSL/graphit/blob/master/autotune/README.md) carefully before trying it out. 
