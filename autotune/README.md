GraphIt Auotuner
=========
The autotuner for GraphIt aims to automatically find the best schedules for a given input algorithm specification and input graph. The autotuner is built on top of OpenTuner to efficiently search for the best combination of schedules (optimizations). **Currently, the autotuner still requires the user to place a label "s1" on the operator that nees to be tuned.**

The autotuner is still very experimental. Please read the instructions below carefully before using it. 

Dependencies
-------------------


Please first install [OpenTuner](https://github.com/jansel/opentuner). GraphIt's autotuner is built on top of OpenTuner. 

Autotune Schedules
-------------------

Currently it is the easiet to use the graphit files under autotune/apps because we need to have hardcoded starting points for the BFS and SSSP. If you want to change the starting point for BFS or SSSP, simply edit bfs_benmark.gt or sssp_benchmark.gt. 


To tune the performance of bfs on the small test graph 4.el, use the following command. `--algo_file` specifies the algorithm file, and `--graph` specifies the input graph to tune on. The autotuner have options that enable/disable parallelization, enable/disable NUMA optimizations (by default we disabled NUMA optimization), and other options such as setting an upper bound on the number of segments to use for configNumSSG. We usually put a **time limt** on tuning with the `--stop-after`  option. Even for programs that run on a large input graph, we can usually finish within 5000 seconds. 
```
#change into the directory for autotuner
cd graphit/auotune

#autotune serial pagerank with 4.el graph for 10 seconds
python graphit_autotuner.py --enable_parallel_tuning 0 --algo_file apps/pagerank_benchmark.gt --graph ../test/graphs/4.el --stop-after 10

#autotune parallel pagerank with 4.el graph for 10 seconds
python graphit_autotuner.py --enable_parallel_tuning 1 --algo_file apps/pagerank_benchmark.gt --graph ../test/graphs/4.el --stop-after 10

```

To see all the options  
```
python graphit_autotuner.py -h
```

The final result will be displayed as a configuration in the standard output, and also in `final_config.json`. One example is shown below. You can then translate that into the scheduling commands. At this time, this translation step still has to be done manually. 
```
('Final Configuration:', {'parallelization': 'serial', 'direction': 'SparsePush', 'numSSG': 5, 'DenseVertexSet': 'boolean-array'})
```

You can also set the serial and parallel compiler used for the C++ files in the graphit_autotune.py file by modifying the variables `serial_compiler` and `par_compiler`.

Currently, the autotuner can not tune the data layout. So the user would need to provide schedules for fusing together different vectors. The user can provide some scheduling commands in a file as a necessary schedule using the `--default_schedule_file`  option. 