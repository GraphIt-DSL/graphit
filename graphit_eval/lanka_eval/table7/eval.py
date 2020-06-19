#!/usr/bin/python 
from collections import defaultdict                                                         
import argparse
import subprocess
import os
import csv

from datetime import datetime

# Configurable threshold to control how much performance difference we are willing to accept
THRESHOLD = 0.1

def check_within(reference, actual):

    """
    Check if the provided number is within the acceptable range from reference.
    """

    if actual >= reference*(1-THRESHOLD) and actual <= reference*(1+THRESHOLD):
        return True 
    return False


def parse_csv(output_file):

    """
    Parses the performance csv file and returns a nested dictionary where
    the first key is graph and the second key is app. It also returns the total lines 
    in the csv file
    """

    result_dictionary = defaultdict(dict)

    reader = csv.reader(open(output_file))
    lines = list(reader)

    for row in lines:
        app, graph, number = row
        result_dictionary[graph][app] = float(number)

    return result_dictionary, len(lines)
            

def check_performance(parse_output, graphs, applications, output_file, reference_file):


    reference_dict, reference_len = parse_csv(reference_file)
    output_dict, output_len = parse_csv(output_file)

    # If we only benchmarked a small subset of applications or graphs, we want to warn the user. 
    if output_len < reference_len:
        print("\033[91mYou are running the performance test on subset of applications!\033[00m")


    passed = True

    for graph in output_dict:
        for app in output_dict[graph]:
            current_number = output_dict[graph][app]
            reference = reference_dict[graph][app]
            passed = check_within(reference, current_number)

            if not passed:
                print("\033[91m{} on {} is too slow. Reference is {} while current is {}\033[00m".format(app, graph, reference, current_number))

    return passed

def main():
    
    # Used to create a new parsed output file
    now = datetime.now()
    current_date =  now.strftime("%m_%d_%Y")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["road", "twitter", "kron", "urand", "web"], help = "graphs to benchmark.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "ds", "pr", "cc", "tc", "bc"],
                        help="applications to benchmark.")
    parser.add_argument('-o', '--output', default='outputs/output_' + current_date + ".csv",
                        help="applications to benchmark. Defaults to all four applications.")
    parser.add_argument('-r', '--reference', default='references/output_06_11_2020.csv',
                        help="applications to benchmark. Defaults to all four applications.")


    args = parser.parse_args()
    graphs_arg = ''
    apps_args = ''

    for graph in args.graphs:
        graphs_arg = graphs_arg + " " + graph 
    print("running benchmarks on graphs: " + graphs_arg)

    for app in args.applications:
        apps_args = apps_args + " " + app
    print("running benchmarks for applications: " + apps_args)

    run_benchmark_cmd = "python2 benchmark.py --graph " + graphs_arg + " --applications " + apps_args
    parse_benchmark_results_cmd = "python2 parse.py --graph " + graphs_arg + " --applications " + apps_args + " --output " + args.output

    out = subprocess.check_call(run_benchmark_cmd, stderr=subprocess.PIPE, shell=True)
    out = subprocess.check_output(parse_benchmark_results_cmd,  stderr=subprocess.PIPE, shell=True)
    print ("Done parsing the run outputs")

    print ("Started running performance test: ")
    result = check_performance(out, args.graphs, args.applications, args.output, args.reference)
    print ("Done running performance test.")
    
    if (result):
        print("\033[92m{}\033[00m" .format("PASSED"))
    else:
        print("\033[91m{}\033[00m" .format("FAILED")) 


if __name__ == "__main__":
    main()
