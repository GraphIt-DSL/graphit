#!/usr/bin/env python                                                           
#
# Autotune schedules for PageRank in the GraphIt language
#                                                                               

import adddeps  # fix sys.path
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

    def compile(self, cfg, id):
        """                                                                          
        Compile a given configuration in parallel                                    
        """
    
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
        GccFlagsTuner.main(argparser.parse_args())
