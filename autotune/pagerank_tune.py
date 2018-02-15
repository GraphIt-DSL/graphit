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




class GraphItPageRankTuner(MeasurementInterface):
    def manipulator(self):
        """                                                                          
        Define the search space by creating a                                        
        ConfigurationManipulator                                                     
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
            EnumParameter('direction', 
                          ['push','pull','hybrid_dense']))

        return manipulator

    def compile(self, cfg, id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """
        
        #write into a schedule file the configuration
        f = open('schedules/default_schedule.gt','r')
        default_schedule_str = f.read()
        f.close()
        new_schedule = default_schedule_str.replace('$direction', cfg['direction'])
        print (new_schedule)
        new_schedule_file_name = 'schedule_' + str(id) 
        print (new_schedule_file_name)
        f1 = open (new_schedule_file_name, 'w')
        f1.write(new_schedule)
        f1.close()
        
        #compile the schedule file along with the original algorithm file
        compile_graphit_cmd = 'python graphitc.py -a ../apps/pagerank_benchmark.gt -f ' + new_schedule_file_name + ' -i ../include/ -l ../build/lib/libgraphitlib.a  -o test.cpp' 
        compile_cpp_cmd = 'g++ -std=c++11 -I ../src/runtime_lib/ -O3  test.cpp -o test'
        print(compile_graphit_cmd)
        print(compile_cpp_cmd)
        self.call_program(compile_graphit_cmd)
        return self.call_program(compile_cpp_cmd)
        
    
    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        """                                                                          
        Run a compile_result from compile() sequentially and return performance      
        """

    def compile_and_run(self, desired_result, input, limit):
        """                                                                          
        Compile and run a given configuration then                                   
        return performance                                                           
        """
        cfg = desired_result.configuration.data
        compile_result = self.compile(cfg, 0)
        return self.run_precompiled(desired_result, input, limit, compile_result, 0)

if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    GraphItPageRankTuner.main(argparser.parse_args())
