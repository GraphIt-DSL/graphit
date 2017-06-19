import os
import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description='compiling graphit files')
    parser.add_argument('-f', dest = 'input_file_name')
    parser.add_argument('-o', dest = 'output_file_name')
    args = parser.parse_args(['-f', 'input'])
    return vars(args)

if __name__ == '__main__':
    args = parseArgs()
    input_file = args['input_file_name']
    output_file = args['input_file_name']

    # read the input file up to the point of schedule:

    # copy these lines to a file called algo.gt

    # generate a file compile.cpp for compiling the algo.gt file

    # generate the default commands

    # add in the scheduling commands

    # compile and execute compile.cpp file to complete the compilation