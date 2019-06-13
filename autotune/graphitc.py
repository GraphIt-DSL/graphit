import argparse
import unittest
import subprocess
import os

def parseArgs():
    parser = argparse.ArgumentParser(description='compiling graphit files')
    parser.add_argument('-f', dest = 'input_file_name')
    parser.add_argument('-o', dest = 'output_file_name')
    parser.add_argument('-a', dest = 'input_algo_file_name')
    parser.add_argument('-i', dest = 'runtime_include_path', default = '../../include/')
    parser.add_argument('-l', dest = 'graphitlib_path', default = '../lib/libgraphitlib.a')
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = parseArgs()
    input_file_name = args['input_file_name']
    output_file_name = args['output_file_name']
    runtime_include_path = args['runtime_include_path']
    graphitlib_path = args['graphitlib_path']
    compile_template_file = "main.cpp"

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

    # generate a file compile.cpp for compiling the algo.gt file
    with open(compile_template_file) as f:
        content = f.readlines()

    # copy over the default file template
    compile_file = open(compile_file_name, 'w')

    for line in content:
        if "insert schedule here" in line:
            #insert the schedule commands here
            for schedule_cmd in schedule_cmd_list:
                compile_file.write(schedule_cmd)

        else:
            compile_file.write(line)

    compile_file.close();

    # compile and execute compile.cpp file to complete the compilation
    #TODO: code here uses very fragile relavtive paths, figure out a better way
    # Maybe setting environment variables
    subprocess.check_call("g++ -g -std=gnu++1y -I {0} {1} -o compile.o {2}".format(runtime_include_path, compile_file_name, graphitlib_path), shell=True)
    subprocess.check_call("./compile.o  -f " + algo_file_name +  " -o " + output_file_name, shell=True)
    #subprocess.check_call("g++ -g -std=c++11 -I ../../src/runtime_lib/  " + output_file_name + " -o test.o", shell=True)
