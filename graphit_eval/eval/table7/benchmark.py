#!/usr/bin/python
import argparse
import subprocess
import os
import shlex
import time
from threading import Timer

framework_app_lookup = {
    "greenmarl": {"pr": "pagerank", "sssp": "sssp", "bfs": "bfs", "cc": "cc"},
    "galois": {"pr": "pagerank", "sssp": "sssp", "bfs": "bfs", "cc": "connectedcomponents"},
    "ligra": {"pr": "PageRank", "sssp": "BellmanFord", "bfs": "BFS", "cc": "Components", "prd": "PageRankDelta", "cf": "CF"},
    "gemini": {"pr": "pagerank", "sssp": "sssp", "bfs": "bfs", "cc": "cc"},
    "graphit": {"pr": "pr", "sssp": "sssp", "bfs": "bfs", "cc": "cc", "prd": "prd", "cf": "cf"},
    "grazelle": {"pr": "pr", "cc": "cc", "bfs": "bfs"},
    "polymer": {"pr": "numa-PageRank", "sssp": "numa-BellmanFord", "bfs": "numa-BFS", "cc": "numa-Components", "prd": "numa-PageRankDelta"}
}

graphit_binary_map = {"socLive" : {"pr":"pagerank_pull", 
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense",
                                   "bfs" :"bfs_hybrid_dense",
                                   "prd" : "pagerankdelta_hybrid_dense"},
                      "twitter" : {"pr":"pagerank_pull_segment",
                                   "sssp" : "sssp_hybrid_denseforward",
                                   "cc" : "cc_hybrid_dense_bitvec_segment",
                                   "bfs" :"bfs_hybrid_dense_bitvec",
                                   "prd" : "pagerankdelta_hybrid_dense_bitvec_numa"}, 
                      "webGraph" : {"pr":"pagerank_pull_segment",
                                    "sssp" : "sssp_hybrid_denseforward",
                                    "cc" : "cc_hybrid_dense_bitvec_segment",
                                    "bfs" :"bfs_hybrid_dense_bitvec_segment",
                                    "prd" : "pagerankdelta_hybrid_dense_bitvec_numa"},
                      "friendster" : {"pr":"pagerank_pull_segment",
                                      "sssp" : "sssp_hybrid_denseforward",
                                      "cc" : "cc_hybrid_dense_bitvec_segment",
                                      "bfs" :"bfs_hybrid_dense_bitvec_segment",
                                      "prd" : "pagerankdelta_hybrid_dense_bitvec_numa"},
                      "road-usad" : {"pr":"pagerank_pull",
                                     "sssp" : "sssp_push_slq",
                                     "cc" : "cc_hybrid_dense",
                                     "bfs" :"bfs_push_slq",
                                     "prd" : "pagerankdelta_hybrid_dense_split"},
                      "netflix"   : {"cf" : "cf_pull_load_balance_segment"},
                      "netflix_2x"   : {"cf" : "cf_pull_load_balance_segment"}
                      }

NUM_THREADS=48
PR_ITERATIONS=20

GreenMarl_PATH = "/data/scratch/baghdadi/green-marl/apps/output_cpp/bin/"
galois_PATH = " /data/scratch/yunming/galois/Galois-2.2.1/build/default/apps/"
ligra_PATH = "/data/scratch/mengjiao/app_binaries/ligra/"
polymer_PATH = "/data/scratch/mengjiao/polymer/"
gemini_PATH = "/data/scratch/mengjiao/GeminiGraph/toolkits/"
graphit_PATH = "/data/scratch/mengjiao/app_binaries/graphit/"
grazelle_PATH = "/data/scratch/mengjiao/app_binaries/grazelle/"
DATA_PATH = "/data/scratch/baghdadi/data3/"
OUTPUT_DIR= "/outputs/"

def get_vertex_count(graph):
    graph_to_count = {"socLive": 4847571, "twitter": 61578415, "webGraph": 101717776,
                    "road-usad": 23947348, "friendster": 124836180, "nlpkkt240": 27993601}
    return graph_to_count[graph]
    
def path_setup(frameworks):
    if not os.path.exists(DATA_PATH + OUTPUT_DIR):
        os.makedirs(DATA_PATH + OUTPUT_DIR)
    for framework in frameworks:
        if not os.path.exists(DATA_PATH + OUTPUT_DIR + framework):
            os.makedirs(DATA_PATH + OUTPUT_DIR + framework);

def get_starting_points(graph):
    """ Use the points with non-zero out degree and don't hang during execution.  """
    if graph != "friendster":
        return ["14", "38", "47", "52", "53", "58", "59", "69", "94", "96"]
    else:
        # friendster takes a long time so use fewer starting points
        return ["101", "286", "16966", "37728", "56030", "155929"]

def get_cmd_grazelle(g, p):
    graph_path = DATA_PATH + g + "/" + g + "_grazelle.bin"
    command = grazelle_PATH + p
    if g == "friendster":
        # run the binary built with 101 as starting point
        command += "101"
    command += " -i "  + graph_path + " -u 0,1"
    if p == "pr":
        command += " -N " + str(PR_ITERATIONS)
    return command

def get_cmd_greenmarl(g, p):
    graph_path = DATA_PATH + g + "/" + g + "_greenmarl.bin"
    args = graph_path + " " + str(NUM_THREADS) + " " + str(PR_ITERATIONS)
    command = GreenMarl_PATH + p + " " + args
    return "numactl -i all " + command

def get_cmd_ligra(g, p, point):
    # netflix graph is only used for collaborative filtering
    if (g == "netflix" and p != "CF"):
        return ""
    # social graphs are not to be used for cf
    if (g != "netflix" and p == "CF"):
        return ""

    args = ""
    if (p == "BellmanFord" or p == "BFS"):
        args += " -r " + point + " " 

    if g == "webGraph" or g == "friendster":
        if g == "friendster":
            # friendster is a symmetric graph
            args += " -s "

        # use binary format for large graphs
        args += " -b "
        if p == "BellmanFord" or p == "CF":
            graph_path = DATA_PATH + g + "/" + g + "-wgh_ligra_bin"
        else:
            graph_path = DATA_PATH + g + "/" + g + "_ligra_bin"
    else:
        if p == "BellmanFord" or p == "CF":
            graph_path = DATA_PATH + g + "/" + g + "_ligra.wadj"
        else:
            graph_path = DATA_PATH + g + "/" + g + "_ligra.adj"

    if p == "PageRank":
        args += " -maxiters 20 "
    elif p == "PageRankDelta":
        args += " -maxiters 10 "
    elif p == "CF":
        args += " -numiter 10 "

    if p == "BellmanFord" and g == "friendster":
        # need LONG option since there are 3.6*2=7.2 billion entries
        p += "_LONG"
                            
    args += graph_path
    command = ligra_PATH + p + " " + args 
    return "numactl -i all " + command

def get_cmd_polymer(g, p, point):
    if p == "numa-BellmanFord":
        graph_path = DATA_PATH + g + "/" + g + "_ligra.wadj"
    else:
        graph_path = DATA_PATH + g + "/" + g + "_ligra.adj"
    command = polymer_PATH + p + " " + graph_path
    if p == "numa-PageRank":
        command += " 20"
    if p == "numa-PageRankDelta":
        command += " 10"
    if p in ["numa-BellmanFord", "numa-BFS"]:
        command += " " + str(point)
    return command

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
    if g in ["twitter", "webGraph", "friendster"]:
        #if p in ["pr", "prd"]:
            #command = "OMP_PLACES=sockets " + command
        #else:
        command = "numactl -i all " + command

        # flexible segment number
        if g == "friendster":
            command += " 30"
        else:
            command += " 16"
    return command

def get_cmd_gemini(g, p, point):
    graph_path = DATA_PATH + g + "/" + g + "_gemini.b"
    if (p == "sssp"):
        graph_path += "wel"
    else:
        graph_path += "el"
    args = graph_path + " " + str(get_vertex_count(g))
    if (p == "sssp" or p == "bfs"):
        args += " " + point
    elif (p == "pagerank"):
        args += " " + str(PR_ITERATIONS)
    command = gemini_PATH + p + " " + args
    return command

def get_cmd_galois(g, p, point):
    if (p == "sssp"):
        extension = "gr"
    else:
        extension = "vgr"
    graph_path = DATA_PATH + g + "/" + g + "_galois." + extension
    special_args = ""
    if ((p == "bfs") or (p == "sssp")):
        special_args += " -startNode=" + point + " "
    if p == "bfs" and g == "road-usad":
        special_args += " -algo=async "
    if p == "connectedcomponents":
        special_args += " -algo=labelProp " # use label propagation rather than union find to be fair
        transpose_graph = DATA_PATH + g + "/" + g + "_galois.tvgr" 
        special_args += " -graphTranspose=" + transpose_graph + " "
    if p == "sssp":
        special_args += " -algo=ligra " # use bellman-ford rather than delta stepping to be fair
        transpose_graph = DATA_PATH + g + "/" + g + "_galois.tgr" 
        special_args += " -graphTranspose=" + transpose_graph + " "
    if (p == "pagerank"):
        transpose_graph = DATA_PATH + g + "/" + g + "_galois.prpull" 
        special_args += " -maxIterations=" + str(PR_ITERATIONS) + " -graphTranspose=" + transpose_graph + " "
    args = graph_path + " -t=" + str(NUM_THREADS) + " " + special_args
    command = galois_PATH + p + "/" + p + " " + args
    return "numactl -i all " + command

def get_cmd(framework, graph, app, point):
    if framework == "greenmarl":
        cmd = get_cmd_greenmarl(graph, app)
    elif framework == "ligra":
        cmd = get_cmd_ligra(graph, app, point)
    elif framework == "galois":
        cmd = get_cmd_galois(graph, app, point)
    elif framework == "graphit":
        cmd = get_cmd_graphit(graph, app, point)
    elif framework == "grazelle":
        cmd = get_cmd_grazelle(graph, app)
    elif framework == "polymer":
        cmd = get_cmd_polymer(graph, app, point)
    else:
        assert framework == "gemini"
        cmd = get_cmd_gemini(graph, app, point)
    return cmd

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--frameworks', nargs='+',
                        default=["galois", "ligra", "graphit", "greenmarl", "gemini"], # "netflix" only used for cf
                        help="frameworks to benchmark")
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["socLive", "road-usad", "twitter", "webGraph", "friendster"],
                        help="graphs to run the applications on. Defaults to all five graphs.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "sssp", "pr", "cc"], # "prd" and "cf" only exists in ligra and graphit
                        help="applications to benchmark. Defaults to all four applications.")
    args = parser.parse_args()

    path_setup(args.frameworks)

    for graph in args.graphs:
        for framework in args.frameworks:
            for generic_app_name in args.applications:
                assert generic_app_name in framework_app_lookup[framework]

                app = framework_app_lookup[framework][generic_app_name]
                log_file_str = DATA_PATH + OUTPUT_DIR + framework + "/" + generic_app_name + "_" + graph + ".txt"
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
