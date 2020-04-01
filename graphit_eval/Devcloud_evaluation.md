# GraphIt Code Generation and Performance Evaluation Guide for Intel Paper Submission

**This instruction assumes that the users have followed the [Getting Started Guide](https://github.com/GraphIt-DSL/graphit/blob/master/README.md ) to set up GraphIt.** The following overview consists of a Step by Step Instructions explaining how to reproduce the performance numbers for both Dataset #1 and Dataset #2. 

## Instruction to access necessary files

After successfully compiling the GraphIt, we can go to evaluation folder:

To access relevant files for dataset #1:

```
  #start from graphit root directory
  cd graphit_eval/
  cd devcloud_eval_dataset1/table7/

```

To access relevant files for dataset #2:

```
  #start from graphit root directory
  cd graphit_eval/
  cd devcloud_eval_dataset2/table7/

```

In each of the folders, you will see something like this:

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

**schedules/** directory contains all the different GraphIt schedules for the above applications. 

**cpps/** directory **will** contain all the C++ files after compiling inside the evaluation directory.

**Makefile** compiles GraphIt applications to binaries. (GraphIt file -> C++ file -> binary file)

**benchmark.py** this script collects the performance numbers for specific applications assuming the files are already compiled. 

**parse.py** this script parses the performance number that **benchmark.py** already collected and outputs numbers in a human readable format. 


## Reproducing results

First, we should be in the respective evaluation directory:

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

This compiles the GraphIt files into binaries that we can directly from benchmark script. We use GCC compiler with Parallel flag to use OPENMP/CILK. After the above step is completed, do:

```
  python2 benchmark.py -a your_application -g your graph

```

This will run the benchmark for a specific application and for a specific graph. For example, you can run BFS on road graph by doing:

```
  python2 benchmark.py -a bfs -g road

```

To run the entire benchmark for all applications and graphs, you can just do:

```
  python2 benchmark.py 

```

To run the entire benchmark on one application, you can do:

```
  python2 benchmark.py -a your_application

```

To run the entire benchmark on one graph, you can do:

```
  python2 benchmark.py -g your_graph

```

To parse the results, you can do:

```
  python2 parse.py -a your_application -g your_graph

```

You can also run the parsing script for all applications and graphs same way was benchmarking script. For example ```python2 parse.py``` would run the parsing script for all applications and graphs. Parser output would be something like this:

```
{'graphit': {'testGraph': {'bfs': 3.3333333333333335e-07, 'pr': 1e-06, 'ds': 5e-07, 'cc': 1e-06, }}}
bfs
testGraph, 3.33333333333e-07
ds
testGraph, 5e-07
pr
testGraph, 1e-06
cc
testGraph, 1e-06

Done parsing the run outputs
```

For the devcloud evaluation, following mappings for the arguments are used:
bfs - Breadth First Search
cc - Connected Components
ds - Single Source Shortest Path
pr - PageRank
bc - Betweenness Centrality
tc - Triangle Counting





