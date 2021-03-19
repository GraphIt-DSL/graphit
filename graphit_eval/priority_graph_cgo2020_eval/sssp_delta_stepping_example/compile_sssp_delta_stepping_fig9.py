#!/usr/local/bin/python

# compiles, prints, and process the generated C++ files 
# assumes that the compiler is built in a directory called build/bin (graphit/build/bin)

import subprocess
import os
import sys

algo_file = "../../../apps/sssp_delta_stepping.gt"
build_directory = "../../../build/bin/"

# relative path from build_directory
input_schedules_path = './schedules/'
sssp_demo_dir = './'

def myprint(message):
    print ("----------------------------------------------------------------")
    print (message)
    print ("----------------------------------------------------------------")

# compile GraphIt and myprint the generated C++ file
def compile_and_myprint(input_schedule_file,  message, cpp_filename):
    # compiles the program with a separate input algorithm file and input schedule file
    # the relative schedule path from build directory 
    myprint ("current directory: " + os.getcwd())
    schedule_file = input_schedules_path + input_schedule_file
    compile_cmd = "python " + build_directory +  "/graphitc.py -a " + algo_file + " -f " + schedule_file + " -o " + cpp_filename
    myprint (compile_cmd)
    subprocess.check_call(compile_cmd, shell=True)
    
    myprint ("SSSP Delta Stepping algo file with schedule %s compiled successfully! " % input_schedule_file)

    myprint (message)
    cat_cmd = "cat " + cpp_filename
    myprint ("Priting the generated c++ file: " + cat_cmd)
    subprocess.check_call(cat_cmd, shell=True)

    myprint ("Generated c++ file saved in: " + cpp_filename)
    cpp_compile_cmd = "g++ -g -std=gnu++1y -I ../../../src/runtime_lib/ " + cpp_filename +  " -o test.o"
    
    myprint (cpp_compile_cmd)
    subprocess.check_call(cpp_compile_cmd, shell=True)
    myprint ("Generated c++ file compiled successfully!")


def eval_schedules():
    myprint("Reproducing schedules in Fig.9 and even more complicated schedules: Generating different SSSP Delta Stepping C++ implemenataion with different schedules. Schedules are stored in the ./schedules directory. The algorithm file (sssp_delta_stepping.gt) is in graphit/apps directiory. ")

    #os.chdir(build_directory)

    # Fig 9(a) Schedule with Lazy Bucket Update, Sparsepush direction
    compile_and_myprint("SparsePush_VertexParallel_Delta2.gt", "SparsePush VertexParallel schedule Fig 9(a)", sssp_demo_dir + "sssp_SparsePush_VertexParallel_schedule.cpp")

    # Fig 9(b) Schedule with Lazy Bucket Update, DensePull direction
    compile_and_myprint("DensePull_VertexParallel_Delta2.gt", "DensePull VertexParallel schedule Fig 9(b)", sssp_demo_dir + "sssp_DensePull_VertexParallel_schedule.cpp")

    
    # Fig 9(c) Schedule with Eager Bucket Update
    compile_and_myprint("priority_update_eager_no_merge.gt", "Eager Bucket Update with SparsePush  Fig 9(c), code is in runtime library OrderedProcessingOperator", sssp_demo_dir + "sssp_eager_schedule.cpp")

    # Even more complicated schedule: with Eager Bucket Fusion Optimization
    compile_and_myprint("priority_update_eager_with_merge.gt", "Eager Bucket Fusion with SparsePush (not included in the figure due to space constraint)", sssp_demo_dir + "pagerankdelta_cache_schedule.cpp")

    myprint("Reproducing Fig 9 Successful! The generated C++ files are stored in the current directory")

    clean_cmd = "rm -rf compile.cpp compile.o compile.o.dSYM test.o test.o.dSYM"
    subprocess.check_call(clean_cmd, shell=True)
    

if __name__ == '__main__':
    eval_schedules()
