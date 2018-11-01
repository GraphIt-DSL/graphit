#!/usr/bin/python                                                          
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["testGraph"], help = "enable graphs with \
socLive, road-usad, twitter, webGraph, friendster.Defaults to the test gra\
ph.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "sssp", "pr", "cc", "prd"],
                        help="applications to benchmark. Defaults to all  applications.")

    args = parser.parse_args()
    graphs_arg = ''
    apps_args = ''

    for graph in args.graphs:
        graphs_arg = graphs_arg + " " + graph 
    print("running benchmarks on graphs: " + graphs_arg)

    for app in args.applications:
        apps_args = apps_args + " " + app
    print("running benchmarks for applications: " + graphs_arg)

    run_benchmark_cmd = "python benchmark.py --graph " + graphs_arg + " --applications " + apps_args
    parse_benchmark_results_cmd = "python parse.py --graph " + graphs_arg + " --applications " + apps_args

    out = subprocess.check_call(run_benchmark_cmd, stderr=subprocess.PIPE, shell=True)
    out = subprocess.check_call(parse_benchmark_results_cmd,  stderr=subprocess.PIPE, shell=True)
    print ("Done parsing the run outputs")

if __name__ == "__main__":
    main()
