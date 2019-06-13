#!/usr/local/bin/python

# compiles, prints, and process the generated C++ files 
# assumes that the compiler is built in a directory called build/bin (graphit/build/bin)

import subprocess
import os
import sys

pagerank_delta_algo_file = "../../apps/pagerankdelta.gt"
build_directory = "../../build/bin"

# relative path from build_directory
input_schedules_path = '../../graphit_eval/pagerankdelta_example/schedules/'
pagerank_delta_demo_dir = '../../graphit_eval/pagerankdelta_example/'

def myprint(message):
    print ("----------------------------------------------------------------")
    print (message)
    print ("----------------------------------------------------------------")

# compile GraphIt and myprint the generated C++ file
def compile_and_myprint(input_schedule_file,  message, cpp_filename):
    # compiles the program with a separate input algorithm file and input schedule file
    # the relative schedule path from build directory 
    os.chdir(build_directory)
    myprint ("current directory: " + os.getcwd())
    algo_file = pagerank_delta_algo_file
    schedule_file = input_schedules_path + input_schedule_file
    compile_cmd = "python graphitc.py -a " + algo_file + " -f " + schedule_file + " -o " + cpp_filename
    myprint (compile_cmd)
    subprocess.check_call(compile_cmd, shell=True)
    
    myprint ("PageRankDelta algo file with schedule %s compiled successfully! " % input_schedule_file)

    myprint (message)
    cat_cmd = "cat " + cpp_filename
    myprint ("Priting the generated c++ file: " + cat_cmd)
    subprocess.check_call(cat_cmd, shell=True)

    myprint ("Generated c++ file saved in: " + cpp_filename)
    cpp_compile_cmd = "g++ -g -std=gnu++1y -I ../../src/runtime_lib/ " + cpp_filename +  " -o test.o"
    
    myprint (cpp_compile_cmd)
    subprocess.check_call(cpp_compile_cmd, shell=True)
    myprint ("Generated c++ file compiled successfully!")


def eval_schedules():
    myprint("Reproducing Fig.6: Generating different PageRankDelta C++ implemenataion with different schedules. Schedules are stored in the ./schedules directory. The algorithm file (pagerankdelta.gt) is in graphit/apps directiory. ")

    # Fig 6(a) The default schedule
    compile_and_myprint("empty_schedule.gt", "the default schedule Fig 6(a)", pagerank_delta_demo_dir + "pagerankdelta_default_schedule.cpp")

    # Fig 6(b) Schedule with Direction Configured
    compile_and_myprint("direction_schedule.gt", "Schedule with Direction Configured Fig 6(b)", pagerank_delta_demo_dir + "pagerankdelta_direction_schedule.cpp")
    
    # Fig 6(c) Schedule with Paralleization Configured
    compile_and_myprint("parallel_schedule.gt", "Schedule with Parallel Configured Fig 6(c)", pagerank_delta_demo_dir + "pagerankdelta_parallel_schedule.cpp")

    # Fig 6(d) Schedule with Datalayout Configured 
    compile_and_myprint("datalayout_schedule.gt", "Schedule with Datalayout Configured Fig 6(d)", pagerank_delta_demo_dir + "pagerankdelta_datalayout_schedule.cpp")

    # Even more complicated schedule: Schedule with Cache Optimization Configured (2 segments)
    compile_and_myprint("cache_schedule.gt", "Schedule with Cache Optimized Configured (not included in the figure due to space constraint)", pagerank_delta_demo_dir + "pagerankdelta_cache_schedule.cpp")

    myprint("Reproducing Fig 6 Successful! The generated C++ files are stored in the current directory")

if __name__ == '__main__':
    eval_schedules()
