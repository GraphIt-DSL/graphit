#!/usr/bin/python                                                          
import argparse
import subprocess
import os

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
    print("running benchmarks for applications: " + graphs_arg)

    run_benchmark_cmd = "python2 benchmark.py --graph " + graphs_arg + " --applications " + apps_args
    parse_benchmark_results_cmd = "python2 parse.py --graph " + graphs_arg + " --applications " + apps_args

    out = subprocess.check_call(run_benchmark_cmd, stderr=subprocess.PIPE, shell=True)
    out = subprocess.check_call(parse_benchmark_results_cmd,  stderr=subprocess.PIPE, shell=True)
    print ("Done parsing the run outputs")

if __name__ == "__main__":
    main()
