Overview for OOPSLA 2018 Artifact Evaluation

**GraphIt - A High-Performance Graph DSL**

The following overview consists of two parts: a Getting Started Guide that contains setup instructions and Step by Step Instructions explaining how to reproduce the paper experiments. 

# Getting Started Guide

The GraphIt Compiler is available as an open source project under the MIT license at [github](https://github.com/yunmingzhang17/graphit) with documentation available at [graphit-lang.org](http://http://graphit-lang.org/). It currently supports Linux and MacOS, but not Windows.

## Set up the virtual machine

For convenience, we provide a Linux VirtualBox VM image with taco pre-installed, as well as the benchmarks we used to evaluate taco in the paper.

Instructions for downloading, installing, and using VirtualBox can be found at [virtualbox.org](http://virtualbox.org). 

Then import the VM using the `Machine -> Add` menu in the VirtualBox application.  When the VM boots, log in with the `graphit` username. The password is `OOPSLA2018`. 

Once you have logged in you will see a directory under the home directory ~/graphit. This directory contains a prebuilt version of GraphIt. 

## Manually download, build, and test taco (optional) 

To evaluate GraphIt on your own machine, simply clone the directory from [the github repoistory](https://github.com/yunmingzhang17/graphit).

###Dependencies

To build GraphIt you need to install
[CMake 3.5.0 or greater](http://www.cmake.org/cmake/resources/software.html). This dependency alone will allow you to build GraphIt and generate high-performance C++ implemenations. Currently, we use Python 2.7 for the end-to-end tests. 

To compile the generated C++ implementations with support for parallleism, you need CILK and OPENMP. One easy way to set up both CILK and OPENMP is to use intel parallel compiler (icpc). The compiler is free for [students](https://software.intel.com/en-us/qualify-for-free-software/student). There are also open source CILK (g++ >= 5.3.0 with support for Cilk Plus), and [OPENMP](https://www.openmp.org/resources/openmp-compilers-tools/) implementations. 

To use NUMA optimizations on multi-socket machines, libnuma needs to be installed (on Ubuntu, sudo apt-get install libnuma-dev). We do note, a good number of optimized implementations do not require enabling NUMA optimizations. You can give GraphIt a try even if you do not have libnuma installed.  

###Build Graphit

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

## Basic Evaluation of GraphIt

###Run Test Programs


Once you have GraphIt set up (through the VM or manually installed), You can run the following test to verify the basic functionalities of GraphIt. 

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

When running `test_with_schedules.py`, commands used for compiling GraphIt files, compiling the generated C++ file, and running the compiled binary file are printed. You can reproduce each test and examine the generated C++ files by typing the printed commands in the shell (make sure you are in the build/bin directory). You can also selectively enable a specific test using the TestSuite commands. We provide examples of enabling a subset of Python tests in the comments of the main function in `test_with_schedules.py`. 

Note when running `test.py`, some error message may be printed during the run that are expected. We have expected to fail tests that print certain error messages. Please check the final output. `test_with_schedules.py` might take a few minutes to run. 

###Compile GraphIt Programs

GraphIt compiler currently generates a C++ output file from the .gt input GraphIt programs. 
To compile an input GraphIt file with schedules in the same file (assuming the build directory is in the root project directory). For now, graphitc.py ONLY works in the build/bin directory.

```
    cd build/bin
    python graphitc.py -f ../../test/input_with_schedules/pagerank_benchmark_cache.gt -o test.cpp
    
```
To compile an input algorithm file and another separate schedule file (some of the test files have hardcoded paths to test inputs, be sure to modify that or change the directory you run the compiled files)

The example below compiles the algorithm file (../../test/input/pagerank.gt), with a separate schedule file (../../test/input_with_schedules/pagerank_pull_parallel.gt)

```
   cd build/bin
   python graphitc.py -a ../../test/input/pagerank_with_filename_arg.gt -f ../../test/input_with_schedules/pagerank_pull_parallel.gt -o test.cpp
```

### Compile and Run Generated C++ Programs

To compile a serial version, you can use reguar g++ with support of c++11 standard to compile the generated C++ file (assuming it is named test.cpp).
 
```
    # assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory
    g++ -std=c++11 -I ../../src/runtime_lib/ -O3 test.cpp  -o test
    ./test ../../test/graphs/4.el
```

To compile a parallel version of the c++ program, you will need both CILK and OPENMP. OPENMP is required for programs using NUMA optimized schedule (configApplyNUMA enabled) and static parallel optimizations (static-vertex-parallel option in configApplyParallelization). All other programs can be compiled with CILK. For analyzing large graphs (e.g., twitter, friendster, webgraph) on NUMA machines, numacl -i all improves the parallel performance. For smaller graphs, such as LiveJournal and Road graphs, not using numactl can be faster. 

```
    # assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory

    # compile and run with CILK
    # icpc
    icpc -std=c++11 -I ../../src/runtime_lib/ -DCILK -O3 test.cpp -o test
    # g++ (gcc) with cilk support
    g++ -std=c++11 -I ../../src/runtime_lib/ -DCILK -fcilkplus -lcilkrts -O3 test.cpp -o test
    # run the compiled binary on a small test graph 4.el
    numactl -i all ./test ../../test/graphs/4.el
    
    # compile and run with OPENMP
    # icpc
    icpc -std=c++11 -I ../../src/runtime_lib/ -DOPENMP -qopenmp -O3 test.cpp -o test
    # g++ (gcc) with openmp support
    g++ -std=c++11 -I ../../src/runtime_lib/ -DOPENMP -fopenmp -O3 test.cpp -o test
    # run the compiled binary on a small test graph 4.el
    numactl -i all ./test ../../test/graphs/4.el

```

You should see some running times printed. The pagerank example files require a commandline argument for the input graph file. If you see a segfault, then it probably means you did not specify an input graph. 


# Step By Step Instructions

