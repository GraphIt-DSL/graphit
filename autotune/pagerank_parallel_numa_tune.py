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



class GraphItPageRankTuner(MeasurementInterface):
    new_schedule_file_name = ""

    def manipulator(self):
        """                                                                          
        Define the search space by creating a                                        
        ConfigurationManipulator                                                     
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
            EnumParameter('direction', 
                          ['SparsePush','DensePull', 'SparsePush-DensePull', 'DensePush-SparsePush']))
        manipulator.add_parameter(EnumParameter('parallelization',['dynamic-vertex-parallel','edge-aware-dynamic-vertex-parallel']))
        manipulator.add_parameter(IntegerParameter('numSSG', 1, 15))
        return manipulator

    #configures parallelization commands
    def write_par_schedule(self, use_evp, new_schedule, direction):
        if use_evp == False:
            new_schedule = new_schedule + "\n    program->configApplyParallelization(\"s1\", \"dynamic-vertex-parallel\");"
        else:
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
        return new_schedule

    def write_numSSG_schedule(self, numSSG, new_schedule, direction):
        # configuring cahce optimization
        if direction == "DensePull" or direction == "SparsePush-DensePull":
            new_schedule = new_schedule + "\n    program->configApplyNumSSG(\"s1\", \"fixed-vertex-count\", " + str(numSSG) + ", \"DensePull\");"
        return new_schedule

    def write_cfg_to_schedule(self, cfg):
        #write into a schedule file the configuration
        direction = cfg['direction']
        numSSG = cfg['numSSG']
        use_evp = False;
        if cfg['parallelization'] == 'edge-aware-dynamic-vertex-parallel':
            use_evp = True;

        f = open('schedules/default_schedule.gt','r')
        default_schedule_str = f.read()
        f.close()

        new_schedule = default_schedule_str.replace('$direction', cfg['direction'])

        new_schedule = self.write_par_schedule(use_evp, new_schedule, direction)
        new_schedule = self.write_numSSG_schedule(numSSG, new_schedule, direction)
        print (cfg)
        print (new_schedule)

        self.new_schedule_file_name = 'schedule_0' 
        print (self.new_schedule_file_name)
        f1 = open (self.new_schedule_file_name, 'w')
        f1.write(new_schedule)
        f1.close()

    def compile(self, cfg, id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """
        #compile the schedule file along with the original algorithm file
        compile_graphit_cmd = 'python graphitc.py -a apps/pagerank_benchmark.gt -f ' + self.new_schedule_file_name + ' -i ../include/ -l ../build/lib/libgraphitlib.a  -o test.cpp' 
        #compile_cpp_cmd = 'g++ -std=c++11  -I ../src/runtime_lib/ -O3  test.cpp -o test'
        compile_cpp_cmd = 'icpc -std=c++11 -DCILK  -I ../src/runtime_lib/ -O3  test.cpp -o test'
        print(compile_graphit_cmd)
        print(compile_cpp_cmd)
        try:
            self.call_program(compile_graphit_cmd)
        except:
            print "fail to compile .gt file"
            exit()
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
        assert compile_result['returncode'] == 0
        try:    
            #run_result = self.call_program('./test ../test/graphs/socLive_gapbs.sg > test.out')
            # run_result = self.call_program('./test ../test/graphs/4.sg > test.out')
            run_result = self.call_program('./test /data/scratch/baghdadi/data3/twitter/twitter_gapbs.sg  > test.out')
            print "run result: " + str(run_result)
            assert run_result['returncode'] == 0
        finally:
            self.call_program('rm test')
            self.call_program('rm test.cpp')
        val = self.parse_running_time();
        self.call_program('rm test.out')
        print "running time: " + str(val)
        return opentuner.resultsdb.models.Result(time=val)
        #return Result(time=run_result['time'])

    def compile_and_run(self, desired_result, input, limit):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        cfg = desired_result.configuration.data

        # converts the configuration into a schedule
        self.write_cfg_to_schedule(cfg)
        
        # this pases in the id 0 for the configuration
        compile_result = self.compile(cfg, 0)
        print "compile_result: " + str(compile_result)
        return self.run_precompiled(desired_result, input, limit, compile_result, 0)


    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print 'Final Configuration:', configuration.data
        self.manipulator().save_to_file(configuration.data,'final_config.json')



if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    GraphItPageRankTuner.main(argparser.parse_args())
