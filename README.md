GraphIt Domain Specific Langauge and Compiler.

Dependencies
===========

To build GraphIt you need to install
[CMake 3.5.0 or greater](http://www.cmake.org/cmake/resources/software.html). This dependency alone will allow you to build GraphIt and generate high-performance C++ implemenations. 

To compile the generated C++ implementations with support for parallleism, you need CILK and OPENMP. One easy way to set up both CILK and OPENMP is to use intel parallel compiler (icpc). The compiler is free for [students](https://software.intel.com/en-us/qualify-for-free-software/student). There are also open source CILK ([Tapir](http://cilk.mit.edu/tapir/)), and [OPENMP](https://www.openmp.org/resources/openmp-compilers-tools/) implementations. 

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

Compile GraphIt Programs
===========
GraphIt compiler currently generates a C++ output file from the .gt input GraphIt programs. 
To compile an input GraphIt file with schedules in the same file (assuming the build directory is in the root project directory). 

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
```

Input Graph Formats
===========

GraphIt reuses [GAPBS input formats](https://github.com/sbeamer/gapbs). Specifically, we have tested with edge list file (.el), weighted edge list file (.wel), binary edge list (.sg), and weighted binary edge list (.wsg) formats. Users can use the converters in GAPBS (GAPBS/src/converter.cc) to convert other graph formats into the supported formats, or convert weighted and unweighted edge list files into their respective binary formats. 

We have provided sample input graph files in the graphit/test/graphs/ directory. The python tests use the sample input files. 

Evaluate GraphIt's Performance
===========

