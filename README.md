GraphIt Domain Specific Langauge and Compiler.

Dependencies
===========

To build GraphIt you need to install
[CMake 3.5.0 or greater](http://www.cmake.org/cmake/resources/software.html). This dependency alone will allow you to build GraphIt and generate high-performance C++ implemenations. Currently, we use Python 2.7 for the end-to-end tests. 

To compile the generated C++ implementations with support for parallleism, you need CILK and OPENMP. One easy way to set up both CILK and OPENMP is to use intel parallel compiler (icpc). The compiler is free for [students](https://software.intel.com/en-us/qualify-for-free-software/student). There are also open source CILK (g++ >= 5.3.0 with support for Cilk Plus), and [OPENMP](https://www.openmp.org/resources/openmp-compilers-tools/) implementations. 

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
Currently, we do require you to name the build directory `build` for the unit tests to work. 
To run the C++ test suite do (all tests should pass):

```
    cd build/bin
    ./graphit_test
```

To run the Python end-to-end test suite:

start at the top level graphit directory cloned from Github, NOT the build directory
(All tests would pass, but some would generate error messages from the g++ compiler. This is expected.)
Currently the project supports Python 2.x and not Python 3.x (the print syntax is different)

```
    cd graphit/test/python
    python test.py
    python test_with_schedules.py
```

When running `test_with_schedules.py`, commands used for compiling GraphIt files, compiling the generated C++ file, and running the compiled binary file are printed. You can reproduce each test and examine the generated C++ files by typing the printed commands in the shell (make sure you are in the build/bin directory). You can also selectively enable a specific test using the TestSuite commands. Examples are given in the comments of the main function in `test_with_schedules.py`. 

Compile GraphIt Programs
===========
GraphIt compiler currently generates a C++ output file from the .gt input GraphIt programs. 
To compile an input GraphIt file with schedules in the same file (assuming the build directory is in the root project directory). For now, graphitc.py ONLY works in the build/bin directory.

```
    cd build/bin
    python graphitc.py -f ../../test/input/simple_vector_sum.gt -o test.cpp
    
```
To compile an input algorithm file and another separate schedule file (some of the test files have hardcoded paths to test inputs, be sure to modify that or change the directory you run the compiled files)

The example below compiles the algorithm file (../../test/input/cc.gt), with a separate schedule file (../../test/input_with_schedules/cc_pull_parallel.gt)

```
   cd build/bin
   python graphitc.py -a ../../test/input/cc.gt -f ../../test/input_with_schedules/cc_pull_parallel.gt -o test.cpp
```

Compile and Run Generated C++ Programs
===========
To compile a serial version, you can use reguar g++ with support of c++11 standard to compile the generated C++ file (assuming it is named test.cpp).
 
```
    # assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory
    g++ -std=c++11 -I ../../src/runtime_lib/ test.cpp  -o -O3 test.o
    ./test.o
```

To compile a parallel version of the c++ program, you will need both CILK and OPENMP. OPENMP is required for programs using NUMA optimized schedule (configApplyNUMA enabled) and static parallel optimizations (static-vertex-parallel option in configApplyParallelization). All other programs can be compiled with CILK. For analyzing large graphs (e.g., twitter, friendster, webgraph) on NUMA machines, numacl -i all improves the parallel performance. For smaller graphs, such as LiveJournal and Road graphs, not using numactl can be faster. 

```
    # assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory

    # compile and run with CILK
    icpc -std=c++11 -I ../../src/runtime_lib/ -DCILK test.cpp -O3 -o  test.o
    numactl -i all ./test.o
    
    # compile and run with OPENMP
    icpc -std=c++11 -I ../../src/runtime_lib/ -DOPENMP -qopenmp -O3 -o test.o
    numactl -i all ./test.o
    
    # to run with NUMA optimizations
    OMP_PLACES=sockets ./test.o 
    
```


Evaluate GraphIt's Performance
===========

The algorithms we used for benchmarking, such as PageRank, PageRankDelta, BFS, Connected Components, Single Source Shortest Paths and Collaborative Filtering are in the **apps** directory.
These files include ONLY the algorithm and NO schedules. You need to use the appropriate schedules for the specific algorithm and input graph to get the best performance. 

In the [arxiv paper](https://arxiv.org/abs/1805.00923) (Table 8), we described the schedules used for each algorithm on each graph on a dual socket system with Intel Xeon E5-2695 v3 CPUs with 12 cores
each for a total of 24 cores and 48 hyper-threads. The system has 128GB of DDR3-1600 memory
and 30 MB last level cache on each socket, and runs with Transparent Huge Pages (THP) enabled. The best schedule for a different machine can be different. You might need to try a few different set of schedules for the best performance. Autotuning is in the works, we will update the instructions with autotuner later. 

In the schedules shown in Table 8, the keyword ’Program’ and the continuation symbol ’->’ are omitted. ’ca’ is the abbreviation for ’configApply’. Note that configApplyNumSSG uses an integer parameter (X) which is dependent on the graph size and the cache size of a system. For example, the complete schedule used for CC on Twitter graph is the following (X is tuned to the cache size)

```
schedule:
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1","dynamic-vertex-parallel");
    program->configApplyNumSSG("s1", "fixed-vertex-count",  X, "DensePull");
```

The **test/input** and **test/input\_with\_schedules** directories contain many examples of the algorithm and schedule files. Use them as references when writing your own schedule.

Input Graph Formats
===========

GraphIt reuses [GAPBS input formats](https://github.com/sbeamer/gapbs). Specifically, we have tested with edge list file (.el), weighted edge list file (.wel), binary edge list (.sg), and weighted binary edge list (.wsg) formats. Users can use the converters in GAPBS (GAPBS/src/converter.cc) to convert other graph formats into the supported formats, or convert weighted and unweighted edge list files into their respective binary formats. 

We have provided sample input graph files in the graphit/test/graphs/ directory. The python tests use the sample input files. 

