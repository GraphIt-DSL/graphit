Overview for OOPSLA 2018 Artifact Evaluation

**GraphIt - A High-Performance Graph DSL**

The following overview consists of two parts: a Getting Started Guide that contains instructions for setting up GraphIt, running tests, compiling and running GraphIt programs; a Step by Step Instructions explaining how to reproduce some tables and figures in the paper.

# Getting Started Guide

The GraphIt compiler is available as an open source project under the MIT license at [github](https://github.com/yunmingzhang17/graphit) with documentation of the language available at [graphit-lang.org](http://graphit-lang.org/). It currently supports Linux and MacOS, but not Windows.

To start, please download and uncompress the `oopsla_artifact.tar.gz` file. The `oopsla18_artifact` directory contains a VM with GraphIt installed, a compressed version of the master branch of GraphIt Github repository and two GraphIt graph files (socLive.sg and socLive.wsg for the Live Journal file). There is also an uncompressed version `oopsla_artifact` in the Dropbox directory in case there is some issue with decompression. We have included a md5 hash of the compressed file in the directory.

The GraphIt github repo and graph files in `oopsla18_artifact` are **not required** to complete the artifact evaluation. They can be used by reviewers who are interested in setting up GraphIt on a more powerful machine to replicate performance numbers from the paper. In addition to `oopsla_artifact.tar.gz`, we have also included an `additional_graphit_graphs` folder that contains larger test graphs. The `additional_graphit_graphs` is also **not necessary for the artifact evaluation** and is very large (30GB).

## Set up the virtual machine

For convenience, we provide a Linux VirtualBox VM image (OOPSLA2018) with GraphIt pre-installed, as well as the benchmarks we used to evaluate GraphIt in the paper. Instructions for downloading, installing, and using VirtualBox can be found at [virtualbox.org](http://virtualbox.org). The virtual machine currently ask for 4GB of memory.

Import the VM using the `Machine -> Add` menu in the VirtualBox application.  When the VM boots, log in with the `graphit` username. The **password is oopsla2018** (lower case). Once you have logged in you will see a directory under the home directory `~/OOPSLA_Artifact/graphit`. This directory contains a prebuilt version of GraphIt.

## Manually download, build, and test GraphIt (optional)

If you choose not to use the GraphIt version already installed in the VM, you can install GraphIt on your own. Simply clone the directory from [the github repository](https://github.com/yunmingzhang17/graphit), or use `graphit-master.zip` (a snapshot of the repository) in the artifact evaluation folder.

### Dependencies

To build GraphIt you need to install
[CMake 3.5.0 or greater](http://www.cmake.org/cmake/resources/software.html). This dependency alone will allow you to build GraphIt and generate high-performance C++ implementations. Currently, we use Python 2.7 for the end-to-end tests.

To compile the generated C++ implementations with support for parallelism, you need Cilk and OpenMP. One easy way to set up both Cilk and OpenMP is to use intel parallel compiler (icpc). The compiler is free for [students](https://software.intel.com/en-us/qualify-for-free-software/student). There are also open source Cilk (g++ >= 5.3.0 with support for Cilk Plus), and [OpenMP](https://www.openmp.org/resources/openmp-compilers-tools/) implementations.

To use NUMA optimizations on multi-socket machines, libnuma needs to be installed (on Ubuntu, sudo apt-get install libnuma-dev). We do note, a good number of optimized implementations do not require enabling NUMA optimizations. You can give GraphIt a try even if you do not have libnuma installed.

### Build GraphIt

To perform an out-of-tree build of GraphIt do:

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

### Run Test Programs


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

When running `test_with_schedules.py`, commands used for compiling GraphIt files, compiling the generated C++ file, and running the compiled binary file are printed. You can reproduce each test and examine the generated C++ files by typing the printed commands in the shell (make sure you are in the `build/bin` directory). You can also selectively enable a specific test using the TestSuite commands. We provide examples of enabling a subset of Python tests in the comments of the main function in `test_with_schedules.py`.

Note when running `test.py`, some error message may be printed during the run and they are **expected**. Please check the final output to see if tests passed. **Running `test.py` and  `test_with_schedules.py` in the VM might take 15-20 minutes**.

### Compile GraphIt Programs

GraphIt compiler currently generates a C++ output file from the .gt input GraphIt programs.
To compile an input GraphIt file with schedules in the same file (assuming the build directory is in the root project directory). For now, graphitc.py ONLY works in the `build/bin` directory.

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

To compile a serial version, you can use g++ with support of c++11 standard to compile the generated C++ file (assuming it is named test.cpp).

```
    #assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory
    g++ -std=c++11 -I ../../src/runtime_lib/ -O3 test.cpp  -o test
    ./test ../../test/graphs/4.el
```

To compile a parallel version of the C++ program, you will need both Cilk and OpenMP. OpenMP is required for programs using NUMA optimized schedule (configApplyNUMA enabled) and static parallel optimizations (static-vertex-parallel option in configApplyParallelization). All other programs can be compiled with Cilk. For analyzing large graphs (e.g., Twitter, friendster, webgraph) on NUMA machines, numactl -i all improves the parallel performance. For smaller graphs, such as LiveJournal and Road graphs, not using numactl can be faster.

```
    #assuming you are still in the bin directory under build/bin. If not, just do cd build/bin from the root of the directory

    #compile and run with Cilk
    #icpc
    icpc -std=c++11 -I ../../src/runtime_lib/ -DCILK -O3 test.cpp -o test
    #g++ (gcc) with cilk support
    g++ -std=c++11 -I ../../src/runtime_lib/ -DCILK -fcilkplus -lcilkrts -O3 test.cpp -o test
    #run the compiled binary on a small test graph 4.el
    numactl -i all ./test ../../test/graphs/4.el

    #compile and run with OpenMP
    #icpc
    icpc -std=c++11 -I ../../src/runtime_lib/ -DOPENMP -qopenmp -O3 test.cpp -o test
    #g++ (gcc) with openmp support
    g++ -std=c++11 -I ../../src/runtime_lib/ -DOPENMP -fopenmp -O3 test.cpp -o test
    #run the compiled binary on a small test graph 4.el
    numactl -i all ./test ../../test/graphs/4.el

```

You should see some running times printed. The pagerank example files require a command-line argument for the input graph file. If you see a segfault, then it probably means you did not specify an input graph.


# Step By Step Instructions

## Reproducing Figure 6
Figure 6 in the paper shows the different C++ code generated by applying different schedules to PageRankDelta. We have build a script to generate the code for PageRankDelta with different schedules and make sure the generated C++ code compiles.

**This script might run for 4-5 minutes as compiling GraphIt code currently is a bit slow. Please wait for a few minutes for the compilation process to finish.**

```
   #start from graphit root directory
  cd  graphit_eval/
  cd pagerankdelta_example
  python compile_pagerankdelta_fig6.py
```

The program should output the information on each schedule, print the generated C++ file to stdout, save the generated file in .cpp files in the directory. The schedules we used are stored in `pagerankdelta_example/schedules`. We added a cache optimized schedule that was not included in the paper due to space constraints. This experiment demonstrates GraphIt's ability to compose together cache, direction, parallelization and data structure optimizations.

## Reproducing Table 7 for GraphIt
Table 7 in the paper shows the performance numbers of GraphIt and other frameworks on 6 applications. Here we provide a script that can produce GraphIt's performance for PageRank, PageRankDelta, Breadth-First Search, Single Source Shortest Paths, and Connected Components. Collaborative Filtering is not included in the script as we do not have the right to distribute the Netflix dataset, but we leave instructions for reviewers to compile collaborative filtering in case Netflix dataset is available. Reviewers can download the other frameworks from their github repositories to replicate the performance of the other frameworks.

### Running GraphIt generated programs

The following commands run the serial version of GraphIt on a small test graph (both the unweighted and weighted versions are in `graphit/graphit_eval/data/testGraph `) that is included in the repository. We have included the generated optimized C++ files for our machine.

```
#start from graphit root directory
cd  graphit_eval/eval/table7

#first compile the generated cpp files
make

#run and benchmark the performance
python table7_graphit.py
```

The script first runs the benchmarks and then saves the outputs to the `graphit_eval/eval/table7/outputs/` directory. And then a separate script parses the outputs to generate the final table of performance in the following form. The application and graph inforamtion are shown on the leftmost column, and running times are shown in the second column in seconds.

```
{'graphit': {'testGraph': {'bfs': 3.3333333333333335e-07, 'pr': 1e-06, 'sssp': 5e-07, 'cc': 1e-06, 'prd': 3e-06}}}
bfs
testGraph, 3.33333333333e-07
sssp
testGraph, 5e-07
pr
testGraph, 1e-06
cc
testGraph, 1e-06
prd
testGraph, 3e-06
Done parsing the run outputs
```
These runs should complete very quickly.

The performance in the VM does not reflect the actual performance because the VM has a single core and has limited memory. This script shows the ability for users / reviewers to replicate the performance on some hardware. Optional instructions on replicating the performance in a physical machine are described in later paragraphs.

### Running on additional graphs (optional)

We have provided a few slightly larger graphs for testing. In the folder we have socLive.sg (unweighted binary Live Journal graph), socLive.wsg (weighted binary Live Journal graph). Outside of the compressed file, we have road graph and Twitter graph. The VM has **insufficient memory** to run socLive graph. We recommend running these graphs on a machine with at least 8 GB memory. Running on Twitter graph can potentially take a significant amount of time if your machine does not have a 100 GB memory and many cores. Running these other graphs with serial C++ implementations can also be potentially very slow. Try to use a parallel implementation if possible.

Below we first show the instructions for running socLive graph.

```
#copy the files to the data directories.
#The directory names have to be socLive, road-usad, twitter as we used hard-coded names in the scripts.

mkdir graphit/graphit_eval/eval/data/socLive
cp socLive.sg graphit/graphit_eval/eval/data/socLive
cp socLive.wsg graphit/graphit_eval/eval/data/socLive

#start from graphit root directory
cd  graphit_eval/eval/table7

#first compile the generated cpp files
make

#run and benchmark the performance
python table7_graphit.py --graph socLive

```

The road graph and Twitter graph need to be named as `road-usad` and `twitter`. We have included some instructions below.

```
#copy the files to the data directories.
#The directory names have to be socLive, road-usad, twitter as we used hard-coded names in the scripts.

mkdir graphit/graphit_eval/eval/data/road-usad
cp road-usad.sg graphit/graphit_eval/eval/data/road-usad
cp road-usad.wsg graphit/graphit_eval/eval/data/road-usad

mkdir graphit/graphit_eval/eval/data/twitter
cp twitter.sg graphit/graphit_eval/eval/data/twitter
cp twitter.wsg graphit/graphit_eval/eval/data/twitter

#start from graphit root directory
cd  graphit_eval/eval/table7

#first compile the generated cpp files
#see the section below for info on parallel builds
make

#run and benchmark the performance on both road and Twitter graphs
python table7_graphit.py --graph road-usad twitter

```


### Running parallel versions and replicating performance (optional)


Here we list the instructions for compiling the generated C++ files using icpc or gcc with Cilk and OpenMP. The user mostly need to define a few variables for the Makefile.

```
#start from graphit root directory
cd  graphit_eval/eval/table7

#compile with icpc if you installed the intel compiler
make ICPC_PAR=1

#compile with gcc with Cilk and OpenMP
make GCC_PAR=1

#run and benchmark the performance
python table7_graphit.py --graph socLive
```

As we mentioned earlier, the VM is not a good place to replicate the performance numbers we reported in the paper. To replicate the performance, you will need to 1) use the parallel versions of the generated C++ programs 2) run them on a machine with similar configurations as ours. We used Intel Xeon E5-2695 v3 CPUs with 12 cores
each for a total of 24 cores and 48 hyper-threads. The system has 128GB of DDR3-1600 memory
and 30 MB last level cache on each socket, and runs with Transparent Huge Pages (THP) enabled. The generated C++ files are also not optimized for single-socket or 4-socket machines.

### Generating, converting and testing graphs (optional)

GraphIt reuses [GAPBS input formats](https://github.com/sbeamer/gapbs). Specifically, we have tested with edge list file (.el), weighted edge list file (.wel), binary edge list (.sg), and weighted binary edge list (.wsg) formats. Users can use the converters in GAPBS (GAPBS/src/converter.cc) to convert other graph formats into the supported formats, or convert weighted and unweighted edge list files into their respective binary formats.

For the additional inputs, you can use the compiled binaries in the `graphit_eval/eval/table7/bin` directory to evaluate the performance. We normally use numactl -i all on graphs with 10M+ vertices. `graphit_eval/eval/table7/benchmark.py` has examples of commands used for running the generated binaries.

To use the script for additional graphs, follow the example of socLive on creating directories in `graphit/graphit_eval/eval/data/`. However, certain graphs have to be named in a certain way in order to use our provided script. For example, road graph and Twitter graph need to be named as `road-usad` and `twitter`. Please take a look at `graphit_eval/eval/table7/benchmark.py` for more details.


### Generating the C++ files from GraphIt programs (optional)

The algorithms we used for benchmarking, such as PageRank, PageRankDelta, BFS, Connected Components, Single Source Shortest Paths and Collaborative Filtering are in the **apps** directory.
These files include ONLY the algorithm and NO schedules. You need to use the appropriate schedules for the specific algorithm and input graph to get the best performance.

In the [arxiv paper](https://arxiv.org/abs/1805.00923) (Table 8), we described the schedules used for each algorithm on each graph on a dual socket system with Intel Xeon E5-2695 v3 CPUs with 12 cores
each for a total of 24 cores and 48 hyper-threads. The system has 128GB of DDR3-1600 memory
and 30 MB last level cache on each socket, and runs with Transparent Huge Pages (THP) enabled. The best schedule for a different machine can be different. You might need to try a few different set of schedules for the best performance.

In the schedules shown in Table 8, the keyword ’Program’ and the continuation symbol ’->’ are omitted. ’ca’ is the abbreviation for ’configApply’. Note that configApplyNumSSG uses an integer parameter (X) which is dependent on the graph size and the cache size of a system. For example, the complete schedule used for CC on Twitter graph is the following (X is tuned to the cache size)

```
schedule:
    program->configApplyDirection("s1", "DensePull")->configApplyParallelization("s1","dynamic-vertex-parallel");
    program->configApplyNumSSG("s1", "fixed-vertex-count",  X, "DensePull");
```

The **test/input** and **test/input\_with\_schedules** directories contain many examples of the algorithm and schedule files. Use them as references when writing your own schedule.
