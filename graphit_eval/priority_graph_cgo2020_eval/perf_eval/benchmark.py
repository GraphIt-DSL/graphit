#!/usr/bin/python
import argparse
import subprocess
import os
import shlex
import time
from threading import Timer

# maps to the specific binary name (except for GraphIt)
# wBFS is just DeltaStepping with delta set to 1

framework_app_lookup = {
    "graphit": {"pr": "pr", "sssp": "sssp", "bfs": "bfs", "cc": "cc", "prd": "prd", "cf": "cf", "ds": "sssp_delta_stepping", "ds_lazy" : "sssp_delta_stepping_lazy" ,"ppsp" : "ppsp_delta_stepping", "astar" : "astar", "wBFS" : "wBFS", "kcore" : "kcore", "setcover":"setcover"}
}

#shared across all frameworks
wBFS_runtime_param_dict = {"testGraph" : 1, "socLive_rand1000" : 1, "road-usad_rand1000" : 1, "road-usad_origweights" : 1, "twitter_rand1000" : 1, "friendster": 1,  "webGraph_rand1000": 1, "friendster_rand1000": 1, "socLive_logn" : 1, "com_orkut_W": 1, "webGraph_logn": 1, "twitter_logn" : 1, "friendster_logn": 1}

# stores the runtime parameter for each application, on each graph for each framework
framework_app_graph_runtime_param_map = {
    "graphit" : {"sssp_delta_stepping": {"testGraph" : 1, "socLive_logn" : 1, "socLive_rand1000" : 100, "road-usad_rand1000" : 8000, "road-usad_origweights" : 40000, "twitter_rand1000" : 4, "twitter_logn" : 1, "com_orkut_W": 1, "com_orkut_rand1000": 8, "webGraph_rand1000":4, "webGraph_logn": 1,"friendster_rand1000": 2, "friendster_logn": 5, "germany":400000, "road-central-usa_origweights":400000, "monaco" : 35000},
                 "sssp_delta_stepping_lazy":{"testGraph" : 1, "socLive_rand1000" : 100, "road-usad_rand1000" : 4096, "road-usad_origweights" : 10000, "twitter_rand1000" : 4, "com_orkut_W": 4, "com_orkut_rand1000": 8, "webGraph_rand1000": 8, "friendster_rand1000": 2, "monaco" : 35000},
                 "ppsp_delta_stepping" : {"testGraph" : 1, "socLive_rand1000" : 50, "road-usad_rand1000" : 8000, "road-usad_origweights" : 40000, "twitter_rand1000" : 4, "com_orkut_W": 1, "com_orkut_rand1000": 8, "webGraph_rand1000":4, "friendster_rand1000": 1, "germany":400000, "road-central-usa_origweights":400000, "monaco" : 35000}, 
                 "astar" : {"germany" : 45000, "massachusetts" : 35000, "monaco" : 35000, "road-usad_origweights" : 40000, "road-central-usa_origweights":400000},
                 "wBFS": wBFS_runtime_param_dict}
}


graphit_twitter_binary_dict = {"pr":"pagerank_pull_numa",
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense_bitvec_segment",
                                   "bfs" :"bfs_hybrid_dense_bitvec_segment",
                                   "prd" : "pagerankdelta_hybrid_dense_bitvec_numa",
                               "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                               "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy",
                               "ppsp_delta_stepping" : "ppsp_delta_stepping_with_merge", 
                               "kcore" : "k_core_const_sum_reduce",
                               "setcover" : "set_cover"}

graphit_web_binary_dict = {"pr":"pagerank_pull_numa",
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense_bitvec_segment",
                                   "bfs" :"bfs_hybrid_dense_bitvec_segment",
                                   "prd" : "pagerankdelta_hybrid_dense_bitvec_numa",
                               "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                               "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy",
                               "ppsp_delta_stepping" : "ppsp_delta_stepping_with_merge",
                               "kcore" : "k_core_const_sum_reduce"}


graphit_socLive_binary_dict = {"pr":"pagerank_pull", 
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense",
                                   "bfs" :"bfs_hybrid_dense",
                                   "prd" : "pagerankdelta_hybrid_dense",
                                   "sssp_delta_stepping" : "sssp_delta_stepping_no_merge",
                               "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy",
                               "ppsp_delta_stepping" : "ppsp_delta_stepping_no_merge",
                               "kcore" : "k_core_const_sum_reduce",
                               "setcover" : "set_cover"
}

graphit_road_binary_dict = {"pr":"pagerank_pull",
                                     "sssp" : "sssp_push_slq",
                                     "cc" : "cc_hybrid_dense",
                                     "bfs" :"bfs_push_slq",
                                     "prd" : "pagerankdelta_hybrid_dense_split",
                            "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                            "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy",
                            "ppsp_delta_stepping" : "ppsp_delta_stepping_with_merge",
                            "astar" : "astar_with_merge",
                            "kcore" : "k_core_const_sum_reduce",
                            "setcover" : "set_cover"}

graphit_binary_map = {"testGraph" : graphit_socLive_binary_dict,
                      "socLive":graphit_socLive_binary_dict,
                      "socLive_rand1000" : graphit_socLive_binary_dict,
                      "socLive_logn" : graphit_socLive_binary_dict,
                      "twitter" : graphit_twitter_binary_dict,
                      "twitter_rand1000" : graphit_twitter_binary_dict,
                      "twitter_logn" : graphit_twitter_binary_dict,
                      "com_orkut_W" : graphit_socLive_binary_dict,
                      "com_orkut_rand1000" : graphit_socLive_binary_dict,
                      "webGraph" : {"pr":"pagerank_pull_numa",
                                    "sssp" : "sssp_hybrid_denseforward_segment",
                                    "cc" : "cc_hybrid_dense_bitvec_segment",
                                    "bfs" :"bfs_hybrid_dense_bitvec_segment",
                                    "prd" : "pagerankdelta_hybrid_dense_bitvec_numa"},
                      #webGraph uses the same binary as twitter graph
                      "webGraph_rand1000" : graphit_web_binary_dict,
                      "webGraph_logn" : graphit_web_binary_dict,
                      "friendster_rand1000" : graphit_twitter_binary_dict,
                      "friendster_logn" : graphit_twitter_binary_dict,
                      "friendster" : {"pr":"pagerank_pull_numa",
                                      "sssp" : "sssp_hybrid_denseforward_segment",
                                      "cc" : "cc_hybrid_dense_bitvec_segment",
                                      "bfs" :"bfs_hybrid_dense_bitvec_segment",
                                      "prd" : "pagerankdelta_hybrid_dense_bitvec_numa"},
                      "road-usad" : graphit_road_binary_dict,
                      "road-usad_rand1000" : graphit_road_binary_dict,
                      "road-usad_origweights" : graphit_road_binary_dict,
                      "netflix"   : {"cf" : "cf_pull_load_balance_segment"},
                      "netflix_2x"   : {"cf" : "cf_pull_load_balance_segment"},
                      "monaco" : graphit_road_binary_dict,
                      "germany" : graphit_road_binary_dict,
                      "massachusetts" : {"sssp" : "sssp_push_slq","astar" : "astar_with_merge"},
                      "road-central-usa_origweights": graphit_road_binary_dict
                      }

NUM_THREADS=48
PR_ITERATIONS=20

graphit_PATH = "./bin/"
DATA_PATH = "~/data/"


def get_vertex_count(graph):
    graph_to_count = {"socLive": 4847571, "twitter": 61578415, "webGraph": 101717776,
                    "road-usad": 23947348, "friendster": 124836180, "nlpkkt240": 27993601}
    return graph_to_count[graph]
    
def path_setup(frameworks, LOG_DIR):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    for framework in frameworks:
        if not os.path.exists(LOG_DIR + framework):
            os.makedirs(LOG_DIR + framework);

def get_starting_points(graph):
    """ Use the points with non-zero out degree and don't hang during execution.  """
    if graph == "germany":
      return ["1087", "2019", "2019", "0", "3478", "134", "13", "47"]
    elif graph == "testGraph":
        return ["1","2"]
    elif graph == "massachusetts":
      return ["0", "19", "101", "0", "1000", "44955", "3432", "23"]
    elif graph == "monaco":
      return ["0", "19", "101", "0", "234", "42", "1209", "47"]
    elif graph in ["com_orkut_W", "com_orkut_rand1000"]:
      return ["149", "149", "149", "34066", "34066", "34066", "2250603", "2250603", "2250603"]
    # elif graph == "webGraph_rand1000":
      # return ["5712", "57123", "234562", "789", "7890000", "326845", "1429214", "17926", "1321806"]
    elif graph not in ["friendster","friendster_rand1000", "friendster_logn"]:
        return ["14", "38", "47", "52", "53", "58", "59", "69", "94", "96"]
    else:
        # friendster takes a long time so use fewer starting points
        return ["101", "286", "16966", "37728", "56030", "155929"]


def get_ending_points(graph):
    if graph == "germany":
      return ["11111111", "1002019", "10002019", "12277374", "7832974", "4728972", "3478", "467823"]
    elif graph == "massachusetts":
      return ["121121", "448950", "111001", "448954", "40000", "48955", "298907", "4215"]
    elif graph == "monaco":
      return ["1111", "1589", "989", "1589", "807", "79", "1470", "48"]
    elif graph in ["com_orkut_W", "com_orkut_rand1000"]:
      return ["2250603", "52", "72789", "13", "2250603", "908977", "1971345", "1", "35"]
    # elif graph == "webGraph_rand1000":
      # return ["1429214", "1", "92798789", "1429214", "1", "43768907", "1430483", "1", "759469"]
    elif graph not in ["friendster", "friendster_rand1000"]:
        return ["69", "47", "52", "205",  "10005", "1000000",  "10005", "7000", "400000", "1000000"]
    else:
        # friendster takes a long time so use fewer ending points
        return ["286", "16966", "37728", "56030", "155929", "101"]

def get_cmd_graphit(g, p, point, dst_point):
    # netflix graph is only used for collaborative filtering
    if (g in ["netflix", "netflix_2x"] and p != "cf"):
        return ""
    # social graphs are not to be used for cf
    if (g not in ["netflix", "netflix_2x"] and p == "cf"):
        return ""

    if (p in ["sssp", "sssp_delta_stepping", "sssp_delta_stepping_lazy", "ppsp_delta_stepping", "cf", "wBFS"]):
        graph_path = DATA_PATH + g + "/" + g + "_gapbs.wsg"
    elif p in ["astar"]:
        graph_path = DATA_PATH + g + "/" + g + ".bin"
    elif p in ["kcore", "setcover"]:
        graph_path = DATA_PATH + g + "/" + g + "_ligra.sadj"
    else:
        graph_path = DATA_PATH + g + "/" + g + "_gapbs.sg"

    args = graph_path
    if p in ["astar", "sssp","bfs","sssp_delta_stepping","sssp_delta_stepping_lazy","ppsp_delta_stepping", "wBFS"]:
        args += " " +  point

    if p in ["astar","ppsp_delta_stepping"]:
        args += " " +  dst_point

    # add runtime parameter
    if p in  ["sssp_delta_stepping", "sssp_delta_stepping_lazy", "ppsp_delta_stepping", "astar", "wBFS"] :
        args += " " + str(framework_app_graph_runtime_param_map["graphit"][p][g]) 

    #wBFS uses the same bianry as sssp_delta_stepping
    if p == "wBFS":
        p = "sssp_delta_stepping"
    command = graphit_PATH + graphit_binary_map[g][p] + " " + args

    if g in ["massachusetts"]:
        command = "taskset -c 0-11 " + command

    if g in ["twitter", "twitter_rand1000", "twitter_logn",  "webGraph", "friendster", "com_orkut_W", "com_orkut_rand1000", "webGraph_rand1000", "webGraph_logn", "friendster_rand1000", "friendster_logn", "socLive_rand1000", "socLive_logn", "road-central-usa_origweights", "road-usad_origweights", "germany"]:
        command = "numactl -i all " + command

        # flexible segment number
        if g == "friendster":
            command += " 30"
        else:
            command += " 16"
    return command

#potentially providing both a source and destination point to each framework

def get_cmd(framework, graph, app, src_point, dst_point = 0):
    if framework == "graphit":
        cmd = get_cmd_graphit(graph, app,  src_point, dst_point)
    else:
        print("unsupported framework: " + framework)
    return cmd

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--frameworks', nargs='+',
                        default=['graphit'], # "netflix" only used for cf
                        help="frameworks to benchmark: " + ' '.join(["galois", "ligra", "graphit", "greenmarl", "gemini", "gapbs", "julienne"]))


    # graphs include "socLive", "road-usad", "twitter", "webGraph", "friendster", "socLive_rand1000", "twitter_rand1000", "road_rand1000", "germany", "massachusetts", "monaco" . (the unweighted version of rand1000 graphs are the same as the original ones, just symlinks)
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["socLive"],
                        help="graphs to run the applications on. Defaults to all five graphs.")
    # applications include bfs, sssp, pr, cc, cf, ds, astar (delta stepping)
    parser.add_argument('-a', '--applications', nargs='+',
                        default=[], # "prd" and "cf" only exists in ligra and graphit
                        help="applications to benchmark. Choices are bfs, sssp, pr, cc, cf, ds, ppsp, astar.")

    parser.add_argument('-p', '--logPath', default="./benchmark_logs/", help="path to the log directory")


    args = parser.parse_args()
    LOG_DIR = args.logPath + "/"

    path_setup(args.frameworks, LOG_DIR)

    for graph in args.graphs:
        for framework in args.frameworks:
            for generic_app_name in args.applications:
                assert generic_app_name in framework_app_lookup[framework]
                
                app = framework_app_lookup[framework][generic_app_name]
                log_file_str = LOG_DIR + framework + "/" + generic_app_name + "_" + graph + ".txt"
                log_file = open(log_file_str, "w+")
                successes = 0
                starting_points = get_starting_points(graph)
                ending_points = get_ending_points(graph)
                for i in range(len(starting_points)):
                    starting_point = starting_points[i]
                    ending_point = ending_points[i]
                    cmd = get_cmd(framework, graph, app, starting_point, ending_point)
                    if not cmd:
                        break
                    print cmd

                    # setup timeout for executions that hang
                    kill = lambda process: process.kill()
                    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    timer = Timer(600, kill, [out]) # timeout after 10 min
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
                        if generic_app_name in ["pr", "cc", "prd", "cf", "kcore", "setcover"]:
                            # pagerank, cc, prd, and cf can return when they succeeds once.
                            # greenmarl sets starting point internally
                            break;            
                log_file.write("successes=" + str(successes) + "\n")
                log_file.close()


if __name__ == "__main__":
    main()
