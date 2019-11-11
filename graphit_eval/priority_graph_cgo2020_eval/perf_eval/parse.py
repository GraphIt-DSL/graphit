#!/usr/bin/python                                                                 

import argparse
import subprocess
import os
import shlex
import time

#instructions
# to add an application for GraphIt
#    (1)  update the graphit_apps
#    (2)  update parseGraphIt to add support for parsing the timing results
# to enable parsing the results of a framework
#    (1) set the frameworks list to include either "graphit, ligra or greenmarl"
#    (2) set the applications, all of the applications are following graphit apps (mapped into other frameworks)



def parse_result(log_file_name, app, time_key, delimiter, index, strip_end, divider, inner_cnt, time_key_own_line):
    """
    @time_key: used to find the time of execution in the log file
    @delimiter: used to extract the time
    @index: used together with the delimiter
    @strip_end: end index to strip off (e.g. seconds) from the time value
    @divider: divice this value to convert the time in log into seconds
    @inner_cnt: number of runs that an application performs internally
    @time_key_own_line: if the time_key is one its own line
    """

    #print(log_file_name)

    with open(log_file_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    initial_inner_cnt = inner_cnt
    min_time = 10000
    sum_time = 0
    successes = 1
    for i in range(len(content)):
        line = content[i]
        if line.find("successes") != -1:
            successes = int(line.split("=")[1])
        if line.find(time_key) != -1:
            #print "line: " + line
            if time_key_own_line:
                next_line = content[i+1]
                time_str = next_line.strip()
            else:
                time_str = line.split(delimiter)[index]
                #print time_str
            if strip_end:
                time_str = time_str[:strip_end] # strip the (s) at the end
            time = float(time_str) / divider
            if time < min_time:
                min_time = time
            inner_cnt -= 1
            if inner_cnt == 0 and app in ["bfs", "sssp", "bc", "ds", "ds_lazy", "ppsp", "astar", "wBFS"]:
                sum_time += min_time
                min_time = 10000
                inner_cnt = initial_inner_cnt
    if app in ["pr", "cc", "cf", "prd", "kcore", "setcover"]:
        return min_time
    else: # bfs or sssp
        if successes > 0:
            return sum_time / successes
        return 0

def print_normalized_execution(frameworks, apps, graphs, results_dict):
    for app in apps:
        print app
        for graph in graphs:
            perf = graph + ", "
            first = True
            graphit_time = results_dict["graphit"][graph][graphit_apps[app_idx]]
            for framework in frameworks:
                if first:
                    first = False
                else:
                    perf += ", "
                perf +=  str(results_dict[framework][graph][app]/graphit_time)
            print perf

def print_absolute_execution(frameworks, apps, graphs, results_dict):
    for app in apps:
        print app
        for graph in graphs:
            perf = graph + ", "
            first = True
            for framework in frameworks:
                if first:
                    first = False
                else:
                    perf += ", "
                perf +=  str(results_dict[framework][graph][app])
            print perf

def print_absolute_execution_by_framework(frameworks, apps, graphs, results_dict, delimiter=",", use_three_decimal=True):
    for app in apps:
        print app
        for framework in frameworks:
            perf = framework + " " + delimiter + " " 
            first = True
            for graph in graphs:
                if first:
                    first = False
                else :
                    perf += " " + delimiter + " "
                if (use_three_decimal):
                    perf += str(round(results_dict[framework][graph][app],3))
                else:
                    perf += str(results_dict[framework][graph][app])
            print perf
            
def print_framework_names(frameworks):
    print (" -------------------" )
    column_str = "frameworks: "
    for framework in frameworks:
        column_str = column_str +  framework + " "
    print column_str
    print (" -------------------" )

def print_graph_names(graphs, delimiter=","):
    print (" -------------------" )
    column_str = "graphs: "
    first = True
    for graph in graphs:
        if first:
            first = False
            column_str = column_str +  graph + " "
        else:
            column_str += delimiter + " " + graph + " "
    print column_str
    print (" -------------------" )
    

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--frameworks', nargs='+',
                        default=["graphit"], # "netflix" only used for cf
                        help="frameworks to parse")
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["socLive", "road-usad", "twitter", "webGraph", "friendster"],
                        help="graphs to parse. Defaults to all five graphs.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "sssp", "pr", "cc"], # "prd" and "cf" only exists in ligra and graphit
                        help="applications to parse. Defaults to all four applications.")

    parser.add_argument('-p', '--logPath', default="./benchmark_logs/", help="path to the log directory")
    args = parser.parse_args()

    # time_key, delimiter, index, strip_end, divider, inner_cnt, time_key_own_line
    parse_args = {
                  "graphit": ["elapsed time", "", None, None, 1, 10, True]
    }

    LOG_PATH = args.logPath + "/"

    results = {}
    for framework in args.frameworks:
        results[framework] = {}
        for g in args.graphs:
            results[framework][g] = {}
            for app in args.applications:
                log_file_name = LOG_PATH + framework + "/" + app + "_" + g + ".txt"
                runtime = parse_result(log_file_name, app, *parse_args[framework])
                results[framework][g][app] = runtime
    print results

    print_framework_names(args.frameworks)
    #print_normalized_execution(args.frameworks, args.applications, args.graphs, results)
    print_absolute_execution(args.frameworks, args.applications, args.graphs, results)

    #print_graph_names(args.graphs, "&")
    #print_absolute_execution_by_framework(args.frameworks, args.applications, args.graphs, results, "&")

    
if __name__ == "__main__":
    main()

