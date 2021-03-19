# GraphIt Performance Checker Guide


### General Information


After successfully compiling and build GraphIt, we can go to evaluation folder (we assume you are using lanka to evaluate):


```
  #start from graphit root directory
  cd graphit_eval/
  cd lanka_eval/table7/
```

In the folder, you will see the following structure:

```
  apps/
  outputs/
  references/
  cpps/
  schedules/
  benchmark.py
  Makefile
  parse.py
  ...


```

Everything is pretty much the same as GraphIt evaluation scripts except for the **references/** folder. Inside this folder, you will find the reference performance numbers from different dates. 


### Running the script 

First, we should be in the respective evaluation directory:

```
  #start from graphit root directory
  cd graphit_eval/
  cd lanka_eval/table7/
```

Then, do:

```
  make clean
  make GCC_PAR=1 
```

This compiles the GraphIt files into binaries that we can directly from benchmark script. We use GCC compiler with Parallel flag to use OPENMP/CILK.



To run a specific benchmark, use eval.py. On top of accepting specific graph and application, you can also provide the output file that we store the result of the current run and the reference file that we can compare the result.  

```
  python2 eval.py -a application -g graph -o outputs/<output_file> -r references/<reference_file>
```

For example, you can run BFS on road graph by doing:

```
  python2 eval.py -a bfs -g road -o outputs/bfs.csv -r references/output_06_11_2020.csv
```

This script will do the following:
* Run the BFS on road graph 
* Parses and stores the result in outputs/bfs.csv file.
* Uses the reference file and the output file to check if the performance still remains reasonable.

If the output flag is not given, the script automatically create **outputs/output<current_date>** file to store the current run. 
If the reference flag is not given, the script uses **references/output_06_11_2020.csv** as the default reference numbers. 

To run the entire benchmark for all applications and graphs, you can just do:

```
  python2 eval.py
```


### File Format 

We store the results from each run in CSV file. Each line has 3 values (application, graph, number). For example, it might look like:
```
bfs,road,0.263924171875
bfs,urand,0.65382009375
bfs,twitter,0.27707725
```




