#!/usr/bin/python 
from collections import defaultdict                                                         
import argparse
import subprocess
import os

THRESHOLD = 0.1

LANKA_NUMBERS = {'kron': {'pr': 13.2738, 'bc': 16.5469875, 'cc': 5.64460125, 'tc': 275.2006666666667, 'bfs': 0.41763739062500005, 'ds': 5.777376093750001}, 
                 'web': {'pr': 3.84271, 'bc': 4.796699500000001, 'cc': 1.3919287500000002, 'tc': 17.6009,'bfs': 0.49045709375, 'ds': 0.8411919843750002}, 
                 'twitter': {'pr': 8.39282, 'bc': 6.2741637500000005, 'cc': 1.6267125, 'tc': 60.25676666666667, 'bfs': 0.27692792187500004, 'ds': 2.3435567187499995}, 
                 'urand': {'pr': 14.3377, 'bc': 20.44584375, 'cc': 6.789328125, 'tc': 20.325666666666667, 'bfs': 0.651329109375, 'ds': 7.80777546875}, 
                 'road': {'pr': 0.519829, 'bc': 4.590365625, 'cc': 13.033706249999998, 'tc': 0.050794, 'bfs': 0.263258859375, 'ds': 0.22250728125000005}}

def check_within(reference, actual):

    if actual >= reference*(1-THRESHOLD) and actual <= reference*(1+THRESHOLD):
        return True 
    return False

def check_performance(parse_output, graphs, applications):

    # number of applications plus the title which is a graph name
    num_lines_per_graph = 1 + len(graphs)
    
    # we need to extract the last num_total from parse output
    num_total = len(applications)*num_lines_per_graph
    numbers = (parse_output.split("\n")[:-1])[-num_total:]

    results = defaultdict(dict)
    current_application = None 
    for line in numbers:
        line = line.strip()
        if line in applications:
            current_application = line 
        #it must be graph runtime
        else:
            graph, performance = line.split(",")
            results[graph][current_application] = float(performance)

    is_successful = True
    #start checking results
    for graph in results:
        for app in results[graph]:
            lanka_reference = LANKA_NUMBERS[graph][app]
            current_number = results[graph][app]
            is_acceptable  = check_within(lanka_reference, current_number)
            if not is_acceptable:
                print("\033[91m {} on {} is too slow. Reference is {} while current is {}\033[00m".format(app, graph, lanka_reference, current_number))
                is_successful = False
    return is_successful





def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["road", "twitter", "kron", "urand", "web"], help = "graphs to benchmark.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "ds", "pr", "cc", "tc", "bc"],
                        help="applications to benchmark.")

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
    parse_benchmark_results_cmd = "python2 parse.py --graph " + graphs_arg + " --applications " + apps_args

    out = subprocess.check_call(run_benchmark_cmd, stderr=subprocess.PIPE, shell=True)
    out = subprocess.check_output(parse_benchmark_results_cmd,  stderr=subprocess.PIPE, shell=True)
    print ("Done parsing the run outputs")

    print ("Started running performance test: ")
    result = check_performance(out, args.graphs, args.applications)
    print ("Done running performance test.")
    
    if (result):
        print("\033[92m {}\033[00m" .format("PASSED"))
    else:
        print("\033[91m {}\033[00m" .format("FAILED")) 


if __name__ == "__main__":
    main()
