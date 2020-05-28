# GraphIt Code Generation and Performance Evaluation Guide for Intel Paper Submission using GAP Benchmark Suite by Scott Beamer

The following overview consists of a Step by Step Instructions explaining how to reproduce the performance numbers for both Dataset #1 and Dataset #2. 

## Build GraphIt
**This instruction assumes that the users have followed the [Getting Started Guide](https://github.com/GraphIt-DSL/graphit/blob/master/README.md ) to set up GraphIt.** 


## Replicating Dataset 1 Number
After successfully compiling and build GraphIt, we can go to evaluation folder:


```
  #start from graphit root directory
  cd graphit_eval/
  cd devcloud_eval_dataset1/table7/
```

In the folder, you will see the following structure:

```
  apps/
  outputs/
  cpps/
  schedules/
  benchmark.py
  Makefile
  parse.py
  ...


```

**apps/** directory contains all the GraphIt applications without the schedules. 

**schedules/** directory contains all the different GraphIt schedules for the above applications. Not all schedules are used. 

**cpps/** directory **will** contain all the C++ files after compiling inside the evaluation directory.

**Makefile** compiles GraphIt applications to binaries. (GraphIt file -> C++ file -> binary file)

**eval.py** calls benchmark.py to run the experiments  and then parse.py to parse the results into a table form.

**benchmark.py** this script collects the performance numbers for specific applications assuming the files are already compiled. 

**parse.py** this script parses the performance number that **benchmark.py** already collected and outputs numbers in a human readable format. 


### Reproducing results

First, we should be in the respective evaluation directory:

```
  #start from graphit root directory
  cd graphit_eval/
  cd devcloud_eval_dataset1/table7/
```

Then, do:

```
  make clean
  make GCC_PAR=1 
```

This compiles the GraphIt files into binaries that we can directly from benchmark script. We use GCC compiler with Parallel flag to use OPENMP/CILK.



To run the benchmark for dataset#1, set the following variables

```
export OMP_NUM_THREADS=32; export GOMP_CPU_AFFINITY="0-31"; export CILK_NWORKERS=32; export KMP_AFFINITY="verbose,explicit,proclist=[0-31]"
```

To run a specific benchmark, use eval.py. This script runs **one binary per application for all graphs**. The binaries we used will be printed and can be found within **benchmark.py**. 

```
  python2 eval.py -a application -g graph
```

This will run the benchmark for a specific application and for a specific graph. For example, you can run BFS on road graph by doing:

```
  python2 eval.py -a bfs -g road
```

This should give the following output with running time in seconds
```
{'graphit': {'testGraph': {'bfs': 3.3333333333333335e-07, 'pr': 1e-06, 'ds': 5e-07, 'cc': 1e-06, }}}
bfs
road, 1.23381740625
ds
road, 0.284026953125
pr
road, 0.42012
cc
road, 36.21985625
tc
road, 0.0620836666667
bc
road, 5.353310625
Done parsing the run outputs
```

To run the entire benchmark for all applications and graphs, you can just do:

```
  python2 eval.py 
```

To run the entire benchmark on one application, you can do:

```
  python2 eval.py -a application
```

To run the entire all applications on one graph, you can do:

```
  python2 eval.py -g graph
```

The eval.py script print out the commands used. You can use that to rerun a specific benchmark. 


### (OPTIONAL) Parse Results Later

All results are stored as logs in the outputs directory. As a result, you can parse them separately later.

```
  python2 parse.py -a application -g graph
```

You can also run the parsing script for all applications and graphs same way was benchmarking script. For example ```python2 parse.py``` would run the parsing script for all applications and graphs. Parser output would be something like this:

### All Applications and Graphs

Below we list the applications and graphs For the devcloud evaluation

**_Applications_ (GAP Benchmark Suite)**

bfs - Breadth First Search

cc - Connected Components

ds - Single Source Shortest Path with Delta Stepping

pr - PageRank  

bc - Betweenness Centrality 

tc - Triangle Counting

**_Graphs_ (GAP Benchmark Suite)**

road - USA RoadNetworks

kron - Kron generated graph of scale 27

urand - Uniform Random graph of scale 27

twitter - Twitter social netowrks

web - Web graph


## Replicating Dataset2

The instructions are mostly the same as Relicating Dataset#1 results.
The main difference is that we don't limit the number of threads in this case. However, there are a few data points that are faster with limiting the threadcount as in Dataset#1 (BFS on road, BC on road). In these cases, we just use the numbers generated from Dataset#1.

First, we will go into dataset2 directory:

```
  #start from graphit root directory
  cd graphit_eval/
  cd devcloud_eval_dataset2/table7/
```

Then, do:

```
  make clean
  make GCC_PAR=1 
```

The rest of the instructions is the same as benchmarking dataset1 using eval.py.
To run a specific benchmark, use eval.py. This script runs **more than one binary for each application, and the binaries are potentially specialized for the specific graph**. The schedules we used to generate each binary can be found in **benchmark.py**.

To run a specific test

```
  python2 eval.py -a application -g graph
```

To run all the tests
```
  python2 eval.py
```


