#!/usr/bin/env python                                                           
#
# Autotune schedules for PageRank in the GraphIt language
#                                                                               

# import adddeps  # fix sys.path
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from sys import exit
import argparse

py_graphitc_file = "../build/bin/graphitc.py"
serial_compiler = "g++"

#if using icpc for par_compiler, the compilation flags for CILK and OpenMP needs to be changed
par_compiler = "g++"

class GraphItTuner(MeasurementInterface):
    new_schedule_file_name = ''
    # a flag for testing if NUMA-aware schedule is specified
    use_NUMA = False
    use_eager_update = False
    
    # this flag is for testing on machine without NUMA library support
    # this would simply not tune NUMA-aware schedules
    enable_NUMA_tuning = True
    
    # this would simply not tune parallelization related schedules 
    # for machines without CILK or openmp support 
    enable_parallel_tuning = True
    
    enable_denseVertexSet_tuning = True


    def manipulator(self):
        """                                                                          
        Define the search space by creating a                                        
        ConfigurationManipulator                                                     
        """

        # set the global flags needed for printing schedules
        if self.args.enable_NUMA_tuning == 0:
            self.enable_NUMA_tuning = False
        if self.args.enable_parallel_tuning == 0:
            self.enable_parallel_tuning = False
        if self.args.enable_denseVertexSet_tuning == 0:
            self.enable_denseVertexSet_tuning = False


        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
            EnumParameter('direction', 
                          ['SparsePush','DensePull', 'SparsePush-DensePull', 'DensePush-SparsePush']))

        #'edge-aware-dynamic-vertex-parallel' not supported with the latest g++ cilk implementation
        if self.enable_parallel_tuning:
            manipulator.add_parameter(EnumParameter('parallelization',['dynamic-vertex-parallel'])) 
        else:
            manipulator.add_parameter(EnumParameter('parallelization', ['serial']))

        manipulator.add_parameter(IntegerParameter('numSSG', 1, self.args.max_num_segments))
        
        if self.enable_NUMA_tuning:
            manipulator.add_parameter(EnumParameter('NUMA',['serial','static-parallel']))

        if self.enable_denseVertexSet_tuning:
            manipulator.add_parameter(EnumParameter('DenseVertexSet', ['boolean-array', 'bitvector']))

        # adding new parameters for PriorityGraph (Ordered GraphIt) 
        manipulator.add_parameter(IntegerParameter('delta', 1, self.args.max_delta))

        manipulator.add_parameter(
            EnumParameter('bucket_update_strategy', 
                          ['eager_priority_update','eager_priority_update_with_merge', 'lazy_priority_update']))

        return manipulator

    #configures parallelization commands
    def write_par_schedule(self, cfg, new_schedule, direction):
        use_evp = False;

        if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
            use_evp = True   

        if use_evp == False or self.use_NUMA == True:
            # if don't use edge-aware parallel (vertex-parallel)
            # edge-parallel don't work with NUMA (use vertex-parallel when NUMA is enabled) 
            if cfg['parallelization'] == 'serial': 
                new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"serial\");"
            else:
                # if NUMA is used, then we only use dynamic-vertex-parallel as edge-aware-vertex-parallel do not support NUMA yet
                new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
        elif use_evp == True and self.use_NUMA == False:   
            #use_evp is True
            if direction == "DensePull": 
                # edge-aware-dynamic-vertex-parallel is only supported for the DensePull direction
                new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024, \"DensePull\");"

            elif direction == "SparsePush-DensePull":
                # For now, only the DensePull direction uses edge-aware-vertex-parallel
                # the SparsePush should still just use the vertex-parallel methodx
                new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"edge-aware-dynamic-vertex-parallel\",1024,  \"DensePull\");"

                new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\",1024,  \"SparsePush\");"

            else:
                #use_evp for SparsePush, DensePush-SparsePush should not make a difference
                new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
        else:
            print ("Error in writing parallel schedule")
            exit()
        return new_schedule

    def write_numSSG_schedule(self, numSSG, new_schedule, direction):
        # No need to insert for a single SSG
        if numSSG == 0 or numSSG ==1:
            return new_schedule
        # configuring cache optimization for DensePull direction
        if direction == "DensePull" or direction == "SparsePush-DensePull":
            new_schedule = new_schedule + "\n    program->configApplyNumSSG(\"s1\", \"fixed-vertex-count\", " + str(numSSG) + ", \"DensePull\");"
        return new_schedule

    def write_delta_schedule(self, delta, new_schedule):
        new_schedule = new_schedule + "\n    program->configApplyPriorityUpdateDelta(\"s1\", " + str(delta) + " );"
        return new_schedule

    def write_bucket_update_schedule(self, bucket_update_strategy, new_schedule):
        new_schedules = new_schedule + "\n    program->configApplyPriorityUpdate(\"s1\", \"" + bucket_update_strategy + "\" );"
        return new_schedules

    def write_NUMA_schedule(self,  new_schedule, direction):
        # configuring NUMA optimization for DensePull direction
        if self.use_NUMA:
            if direction == "DensePull" or direction == "SparsePush-DensePull":
                new_schedule = new_schedule + "\n    program->configApplyNUMA(\"s1\", \"static-parallel\" , \"DensePull\");"
        return new_schedule

    def write_denseVertexSet_schedule(self, enable_pull_bitvector, new_schedule, direction):
        # for now, we only use this for the src vertexset in the DensePull direciton
        if direction == "DensePull" or direction == "SparsePush-DensePull":
            if enable_pull_bitvector:
                new_schedule = new_schedule + "\n    program->configApplyDenseVertexSet(\"s1\",\"bitvector\", \"src-vertexset\", \"DensePull\");"
        return new_schedule

    def write_cfg_to_schedule(self, cfg):
        #write into a schedule file the configuration
        direction = cfg['direction']
        numSSG = cfg['numSSG']
        delta = cfg['delta']
        bucket_update_strategy = cfg['bucket_update_strategy']

        new_schedule = ""
        direction_schedule_str = "\n    program->configApplyDirection(\"s1\", \"$direction\");" 
        if self.args.default_schedule_file != "":
            f = open(self.args.default_schedule_file,'r')
            default_schedule_str = f.read()
            f.close()
        else:
            default_schedule_str = "schedule: "
        
            
        #eager only works with SparsePush for now
        if bucket_update_strategy == 'eager_priority_update':
            new_schedule = default_schedule_str + direction_schedule_str.replace('$direction', 'SparsePush')
        else:
            new_schedule = default_schedule_str + direction_schedule_str.replace('$direction', cfg['direction'])

        new_schedule = self.write_par_schedule(cfg, new_schedule, direction)
        new_schedule = self.write_numSSG_schedule(numSSG, new_schedule, direction)
        new_schedule = self.write_NUMA_schedule(new_schedule, direction)
        new_schedule = self.write_delta_schedule(delta, new_schedule)
        new_schedule = self.write_bucket_update_schedule(bucket_update_strategy, new_schedule)

        use_bitvector = False
        if cfg['DenseVertexSet'] == 'bitvector':
            use_bitvector = True
        new_schedule = self.write_denseVertexSet_schedule(use_bitvector, new_schedule, direction)


        print (cfg)
        print (new_schedule)

        self.new_schedule_file_name = 'schedule_0' 
        print (self.new_schedule_file_name)
        f1 = open (self.new_schedule_file_name, 'w')
        f1.write(new_schedule)
        f1.close()

    def compile(self, cfg,  id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """


        #compile the schedule file along with the original algorithm file
        compile_graphit_cmd = 'python ' + py_graphitc_file +  ' -a  {algo_file} -f {schedule_file} -i ../include/ -l ../build/lib/libgraphitlib.a  -o test.cpp'.format(algo_file=self.args.algo_file, schedule_file=self.new_schedule_file_name) 

        if not self.use_NUMA:
            if not self.enable_parallel_tuning:
                # if parallel icpc compiler is not needed (only tuning serial schedules)
                compile_cpp_cmd = serial_compiler + ' -std=gnu++1y  -I ../src/runtime_lib/ -O3  test.cpp -o test'
            else:
                # if parallel icpc compiler is supported and needed
                compile_cpp_cmd = par_compiler + ' -std=gnu++1y -DCILK -fcilkplus  -I ../src/runtime_lib/ -O3  test.cpp -o test'
        else:
            #add the additional flags for NUMA
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -lnuma -DNUMA -fopenmp -I ../src/runtime_lib/ -O3  test.cpp -o test'

        if self.use_eager_update:
            compile_cpp_cmd = 'g++ -std=gnu++1y -DOPENMP -fopenmp -I ../src/runtime_lib/ -O3  test.cpp -o test'
        

        print(compile_graphit_cmd)
        print(compile_cpp_cmd)
        try:
            self.call_program(compile_graphit_cmd)
        except:
            print ("fail to compile .gt file")
        return self.call_program(compile_cpp_cmd)

    def parse_running_time(self, log_file_name='test.out'):
        """Returns the elapsed time only, from the HPL output file"""

        min_time = 10000

        with open(log_file_name) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        i = 0;
        for line in content:
            if line.find("elapsed time") != -1:
                next_line = content[i+1]
                time_str = next_line.strip()
                time = float(time_str)
                if time < min_time:
                    min_time = time
            i = i+1;

        return min_time

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """

        cfg = desired_result.configuration.data
        
        if compile_result['returncode'] != 0:
            print (str(compile_result))

        assert compile_result['returncode'] == 0
        try:    
            #run_result = self.call_program('./test ../test/graphs/socLive_gapbs.sg > test.out')
            # run_result = self.call_program('./test ../test/graphs/4.sg > test.out')
            if not self.use_NUMA:
                if not self.enable_parallel_tuning:
                    # don't use numactl when running serial
                    run_cmd = './test ' + self.args.graph +  '  > test.out'

                else:
                    # use numactl when running parallel
                    run_cmd = 'numactl -i all ./test ' + self.args.graph +  '  > test.out'
            else:
                run_cmd = 'OMP_PLACES=sockets ./test ' + self.args.graph + '  > test.out'

            print ("run_cmd: " + run_cmd)

            # default value -1 for memory_limit translates into None (no memory upper limit)
            # setting memory limit does not quite work yet
            process_memory_limit = None
            if self.args.memory_limit != -1:
                process_memory_limit = self.args.memory_limit
            # print ("memory limit: " + str(process_memory_limit))
            run_result = self.call_program(run_cmd, limit=self.args.runtime_limit, memory_limit=process_memory_limit)  
        finally:
            self.call_program('rm test')
            self.call_program('rm test.cpp')

        if run_result['timeout'] == True:
            val = self.args.runtime_limit
        else:
            val = self.parse_running_time();
        
        self.call_program('rm test.out')
        print ("run result: " + str(run_result))
        print ("running time: " + str(val))

        if run_result['timeout'] == True:
            print ("Timed out after " + str(self.args.runtime_limit) + " seconds")
            return opentuner.resultsdb.models.Result(time=val)
        elif run_result['returncode'] != 0:
            if self.args.killed_process_report_runtime_limit == 1 and run_result['stderr'] == 'Killed\n':
                print ("process killed " + str(run_result))
                return opentuner.resultsdb.models.Result(time=self.args.runtime_limit)
            else:
                print (str(run_result))
                exit()
        else:
            return opentuner.resultsdb.models.Result(time=val)
            
        


    def compile_and_run(self, desired_result, input, limit):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        print "input graph: " + self.args.graph

        cfg = desired_result.configuration.data

        self.use_NUMA = False;
        # only use NUMA when we are tuning parallel and NUMA schedules
        if self.enable_NUMA_tuning and self.enable_parallel_tuning and cfg['NUMA'] == 'static-parallel':
            if cfg['direction'] == 'DensePull' or cfg['direction'] == 'SparsePush-DensePull':
                if int(cfg['numSSG']) > 1:
                    self.use_NUMA = True;


        if cfg['bucket_update_strategy'] == "eager_priority_update" or cfg['bucket_update_strategy'] == "eager_priority_update_with_merge":
            self.use_eager_update = True
        # converts the configuration into a schedule
        self.write_cfg_to_schedule(cfg)
        
        # this pases in the id 0 for the configuration
        compile_result = self.compile(cfg, 0)
        # print "compile_result: " + str(compile_result)
        return self.run_precompiled(desired_result, input, limit, compile_result, 0)


    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print ('Final Configuration:', configuration.data)
        self.manipulator().save_to_file(configuration.data,'final_config.json')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument('--graph', type=str, default="../test/graphs/4.sg",
                    help='the graph to tune on')
    parser.add_argument('--enable_NUMA_tuning', type=int, default=0, help='enable tuning NUMA-aware schedules. 1 for enable (default), 0 for disable')
    parser.add_argument('--enable_parallel_tuning', type=int, default=1, help='enable tuning paralleliation schedules. 1 for enable (default), 0 for disable')
    parser.add_argument('--enable_denseVertexSet_tuning', type=int, default=1, help='enable tuning denseVertexSet schedules. 1 for enable (default), 0 for disable')
    parser.add_argument('--algo_file', type=str, required=True, help='input algorithm file')
    parser.add_argument('--default_schedule_file', type=str, required=False, default="", help='default schedule file')
    parser.add_argument('--runtime_limit', type=float, default=300, help='a limit on the running time of each program')
    parser.add_argument('--max_num_segments', type=int, default=24, help='maximum number of segments to try for cache and NUMA optimizations')
    parser.add_argument('--max_delta', type=int, default=800000, help='maximum delta used for priority coarsening')
    parser.add_argument('--memory_limit', type=int, default=-1,help='set memory limit on unix based systems [does not quite work yet]')    
    parser.add_argument('--killed_process_report_runtime_limit', type=int, default=0, help='reports runtime_limit when a process is killed by the shell. 0 for disable (default), 1 for enable')
    args = parser.parse_args()
    # pass the argumetns into the tuner
    GraphItTuner.main(args)
    
