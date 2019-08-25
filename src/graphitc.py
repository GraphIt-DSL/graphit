import argparse
import unittest
import subprocess
import os

CXX_COMPILER="${CXX_COMPILER}"
GRAPHIT_BUILD_DIRECTORY="${GRAPHIT_BUILD_DIRECTORY}".strip().rstrip('/')
GRAPHIT_SOURCE_DIRECTORY="${GRAPHIT_SOURCE_DIRECTORY}".strip().rstrip('/')




def parseArgs():
    parser = argparse.ArgumentParser(description='compiling graphit files')
    parser.add_argument('-f', dest = 'input_file_name')
    parser.add_argument('-o', dest = 'output_file_name')
    parser.add_argument('-a', dest = 'input_algo_file_name')
    parser.add_argument('-i', dest = 'runtime_include_path', default = GRAPHIT_SOURCE_DIRECTORY+'/include/')
    parser.add_argument('-l', dest = 'graphitlib_path', default = GRAPHIT_BUILD_DIRECTORY+'/lib/libgraphitlib.a')
    parser.add_argument('-m', dest = 'graphit_pybind_module_name', default = "")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = parseArgs()
    input_file_name = args['input_file_name']
    output_file_name = args['output_file_name']
    runtime_include_path = args['runtime_include_path']
    graphitlib_path = args['graphitlib_path']
    graphit_pybind_module_name = args['graphit_pybind_module_name']

    #check if user supplied a separate algorithm file from the schedule file
    supplied_separate_algo_file = False

    if (args['input_algo_file_name']):
        # use the separate algorithm file if supplied, and use the input_file only for the schedule
        supplied_separate_algo_file = True
        algo_file_name = args['input_algo_file_name']
    else:
        # use the input_file for both the algorithm and schedule
        algo_file_name = 'algo.gt'

    compile_file_name = 'compile.cpp'


    # read the input file
    with open(input_file_name) as f:
        content = f.readlines()

    if not supplied_separate_algo_file:
        # copy lines up to the point of 'schedule:' to 'algo.gt' file
        algo_file = open(algo_file_name, 'w')
    schedule_cmd_list = []
    is_processing_schedule = False

    for line in content:
        if line.startswith("schedule:"):
            is_processing_schedule = True
        elif is_processing_schedule:
            schedule_cmd_list.append(line)
        else:
            if not supplied_separate_algo_file:
                algo_file.write(line)

    if not supplied_separate_algo_file:
        algo_file.close();


    COMPILER_BINARY = ""
    if len(schedule_cmd_list) == 0:
        COMPILER_BINARY=GRAPHIT_BUILD_DIRECTORY+"/bin/graphitc"
    else:
        # generate the schedule file schedule.cpp
        compile_file = open(compile_file_name, 'w')

        compile_file.write("#include <graphit/frontend/high_level_schedule.h>\n")
        compile_file.write("namespace graphit {\n")
        compile_file.write("void user_defined_schedule (graphit::fir::high_level_schedule::ProgramScheduleNode::Ptr program) {\n")
        for schedule_cmd in schedule_cmd_list:
            compile_file.write(schedule_cmd)
        compile_file.write("}\n")
        compile_file.write("}")


        compile_file.close();

        # compile and execute compile.cpp file to complete the compilation
        #TODO: code here uses very fragile relavtive paths, figure out a better way
        # Maybe setting environment variables
        try:
            subprocess.check_call(CXX_COMPILER + " -g -std=c++14 -I {0} {1} -o compile.o {2}".format(runtime_include_path, compile_file_name, graphitlib_path), stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise
        COMPILER_BINARY = "./compile.o"

    try:
        if graphit_pybind_module_name == "":
            subprocess.check_call(COMPILER_BINARY + " -f " + algo_file_name +  " -o " + output_file_name , stderr=subprocess.STDOUT, shell=True)
        else:
            subprocess.check_call(COMPILER_BINARY + " -f " + algo_file_name + " -o " + output_file_name + " -m " + graphit_pybind_module_name, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise

    #subprocess.check_call("g++ -g -std=c++11 -I ../../src/runtime_lib/  " + output_file_name + " -o test.o", shell=True)
    if algo_file_name == "algo.gt" and os.path.exists(algo_file_name):
        os.unlink(algo_file_name)

