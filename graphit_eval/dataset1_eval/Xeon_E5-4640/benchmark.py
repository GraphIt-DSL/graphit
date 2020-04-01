#!/usr/bin/python
import argparse
import subprocess
import os
import shlex
import time
from threading import Timer

use_NUMACTL = True 


framework_app_lookup = {
    "graphit": {"pr": "pr", "sssp": "sssp", "bfs": "bfs", "cc": "cc", "prd": "prd", "cf": "cf"},
}

graphit_binary_map = {"testGraph" : {"pr":"pagerank_pull", 
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense",
                                   "bfs" :"bfs_hybrid_dense",
                                   "prd" : "pagerankdelta_hybrid_dense"},
                      "socLive" : {"pr":"pagerank_pull", 
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense",
                                   "bfs" :"bfs_hybrid_dense",
                                   "prd" : "pagerankdelta_hybrid_dense"},
                      "twitter" : {"pr":"pagerank_pull_segment",
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense_bitvec_segment",
                                   "bfs" :"bfs_hybrid_dense_bitvec",
                                   "prd" : "pagerankdelta_hybrid_dense_bitvec_segment"}, 
                      "webGraph" : {"pr":"pagerank_pull_segment",
                                    "sssp" : "sssp_hybrid_denseforward",
                                    "cc" : "cc_hybrid_dense_bitvec_segment",
                                    "bfs" :"bfs_hybrid_dense_bitvec",
                                    "prd" : "pagerankdelta_hybrid_dense_bitvec_segment"},
                      "friendster" : {"pr":"pagerank_pull_segment",
                                      "sssp" : "sssp_hybrid_denseforward",
                                      "cc" : "cc_hybrid_dense_bitvec_segment",
                                      "bfs" :"bfs_hybrid_dense_bitvec",
                                      "prd" : "pagerankdelta_hybrid_dense_bitvec_segment"},
                      "road-usad" : {"pr":"pagerank_pull",
                                     "sssp" : "sssp_push_slq",
                                     "cc" : "cc_hybrid_dense",
                                     "bfs" :"bfs_push_slq",
                                     "prd" : "pagerankdelta_sparse_push"},
                      "netflix"   : {"cf" : "cf_pull_load_balance_segment"},
                      "netflix_2x"   : {"cf" : "cf_pull_load_balance_segment"}
                      }

NUM_THREADS=48
PR_ITERATIONS=20

graphit_PATH = "./bin/"
DATA_PATH = "../data/"
OUTPUT_DIR= "./outputs/"

def get_vertex_count(graph):
    graph_to_count = {"socLive": 4847571, "twitter": 61578415, "webGraph": 101717776,
                    "road-usad": 23947348, "friendster": 124836180, "nlpkkt240": 27993601}
    return graph_to_count[graph]
    
def path_setup(frameworks):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for framework in frameworks:
        if not os.path.exists(OUTPUT_DIR + framework):
            os.makedirs(OUTPUT_DIR + framework);

def get_starting_points(graph):
    """ Use the points with non-zero out degree and don't hang during execution.  """
    if graph == "testGraph":
        return ["1","2"]
    elif graph != "friendster":
        return ["17", "38", "47", "52", "53", "58", "59", "69", "94", "96"]
    else:
        # friendster takes a long time so use fewer starting points
        return ["101", "286", "16966", "37728", "56030", "155929"]

def get_cmd_graphit(g, p, point):
    # netflix graph is only used for collaborative filtering
    if (g in ["netflix", "netflix_2x"] and p != "cf"):
        return ""
    # social graphs are not to be used for cf
    if (g not in ["netflix", "netflix_2x"] and p == "cf"):
        return ""

    if (p == "sssp" or p == "cf"):
        graph_path = DATA_PATH + g + "/" + g + "_gapbs.wsg"
    else:
        graph_path = DATA_PATH + g + "/" + g + "_gapbs.sg"

    args = graph_path
    if p == "sssp" or p == "bfs":
        args += " " +  point

    command = graphit_PATH + graphit_binary_map[g][p] + " " + args
    #if p in ["pr", "cc", "prd"] and g in ["twitter", "webGraph", "friendster"]:
    if g in ["socLive", "twitter", "webGraph", "friendster"]:
        
        if use_NUMACTL:
            # if NUAMCTL is available
            command = "numactl -i all " + command
        # number of segments, need to be tuned to cache
        # this is tuned to a 20MB LLC in Xeon E5-4640
        if g == "friendster":
            command += " 50"
        else:
            command += " 24"

    if g in ["netflix", "netflix_2x"]:
        if use_NUMACTL:
            command = "numactl -i all " + command

        #set the number of segments (tuned to LLC size)
        if g == "netflix":
            command += " 9"
        else:
            command += " 22"

    return command


def get_cmd(framework, graph, app, point):
    if framework == "graphit":
        cmd = get_cmd_graphit(graph, app, point)
    return cmd

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["testGraph"], help = "enable graphs with socLive, road-usad, twitter, webGraph, friendster.Defaults to the test graph.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "sssp", "pr", "cc", "prd"], 
                        help="applications to benchmark. Defaults to all four applications.")
    
    args = parser.parse_args()




    path_setup(["graphit"])

    for graph in args.graphs:
        for framework in ["graphit"]:
            for generic_app_name in args.applications:
                assert generic_app_name in framework_app_lookup[framework]

                app = framework_app_lookup[framework][generic_app_name]
                log_file_str = OUTPUT_DIR + framework + "/" + generic_app_name + "_" + graph + ".txt"
                log_file = open(log_file_str, "w+")
                successes = 0
                points = get_starting_points(graph)
                for point in points:
                    cmd = get_cmd(framework, graph, app, point)
                    if not cmd:
                        break
                    print cmd

                    # setup timeout for executions that hang
                    kill = lambda process: process.kill()
                    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    timer = Timer(7200, kill, [out]) # timeout after 2 hrs
                    try:
                        timer.start()
                        (output, err) = out.communicate()
                    finally:
                        timer.cancel()

                    if output:
                        successes += 1
                        log_file.write("\n------------------------------------------------\n")
                        log_file.write(time.strftime("%d/%m/%Y-%H:%M:%S"))
                        log_file.write("\n")
                        log_file.write(output)
                        log_file.write("\n------------------------------------------------\n")
                        if generic_app_name in ["pr", "cc", "prd", "cf"] or framework in ["greenmarl", "grazelle"]:
                            # pagerank, cc, prd, and cf can return when they succeeds once.
                            # greenmarl sets starting point internally
                            break;            
                log_file.write("successes=" + str(successes) + "\n")
                log_file.close()


if __name__ == "__main__":
    main()
