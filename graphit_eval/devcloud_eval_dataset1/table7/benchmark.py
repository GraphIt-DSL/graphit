#!/usr/bin/python
import argparse
import subprocess
import os
import shlex
import time
from threading import Timer

use_NUMACTL = True 

framework_app_lookup = {
    "graphit": {"pr": "pr", "sssp": "sssp", "bfs": "bfs", "cc": "cc", "bc":"bc", "tc":"tc", "ds": "sssp_delta_stepping", "ds_lazy" : "sssp_delta_stepping_lazy"},
}

# TODO these numbers might not be optimal
framework_app_graph_runtime_param_map = {
    "graphit" : {"sssp_delta_stepping": {"testGraph" : 1, "twitter" : 2, "web" : 2, "kron": 2, "urand": 2, "road":50000},
                 "sssp_delta_stepping_lazy":{"testGraph" : 1, "twitter" : 10000, "web" : 10000, "kron": 10000, "urand": 10000, "road":10000},
                }
}

#TODO kron and urand have to be tuned. Right now they are just placeholders
# graphit_binary_map = {"testGraph" : {"pr":"pagerank_pull", 
#                                    "sssp" : "sssp_hybrid_denseforward",
#                                    "cc" : "cc_hybrid_dense",
#                                    "bfs" :"bfs_hybrid_dense",
#                                    "bc" : "bc_SparsePushDensePull_bitvector",
#                                    "tc" : "tc_hiroshi",
#                                    "sssp_delta_stepping" : "sssp_delta_stepping_no_merge",
#                                    "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
#                        "twitter" : {"pr":"pagerank_pull_segment",
#                                     "sssp" : "sssp_hybrid_denseforward",
#                                     "cc" : "cc_hybrid_dense_bitvec_segment",
#                                     "bfs" :"bfs_hybrid_dense_bitvec",
#                                     "bc" : "bc_SparsePushDensePull_bitvector",
#                                     "tc": "tc_hiroshi",
#                                     "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
#                                     "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
#                         "web" : {"pr":"pagerank_pull_segment",
#                                   "sssp" : "sssp_hybrid_denseforward",
#                                   "cc" : "cc_hybrid_dense_bitvec_segment",
#                                   "bfs" :"bfs_hybrid_dense_bitvec",
#                                   "bc" : "bc_SparsePushDensePull_bitvector",
#                                   "tc": "tc_hiroshi",
#                                   "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
#                                   "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
#                         "kron" : {"pr":"pagerank_pull_segment",
#                                   "sssp" : "sssp_hybrid_denseforward",
#                                   "cc" : "cc_hybrid_dense_bitvec_segment",
#                                   "bfs" :"bfs_hybrid_dense_bitvec",
#                                   "bc" : "bc_SparsePushDensePull_bitvector",
#                                   "tc": "tc_hiroshi",
#                                   "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
#                                   "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
#                         "urand" : {"pr":"pagerank_pull_segment",
#                                   "sssp" : "sssp_hybrid_denseforward",
#                                   "cc" : "cc_hybrid_dense_bitvec_segment",
#                                   "bfs" :"bfs_hybrid_dense_bitvec",
#                                   "bc" : "bc_SparsePushDensePull_bitvector",
#                                   "tc": "tc_hiroshi",
#                                   "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
#                                   "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},

#                         "road" : {"pr":"pagerank_pull_segment",
#                                   "sssp" : "sssp_hybrid_denseforward",
#                                   "cc" : "cc_hybrid_dense_bitvec_segment",
#                                   "bfs" :"bfs_hybrid_dense_bitvec",
#                                   "bc" : "bc_SparsePushDensePull_bitvector",
#                                   "tc": "tc_hiroshi",
#                                   "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
#                                   "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
#                       }

graphit_binary_map = {"testGraph" : {"pr":"pagerank_pull", 
                                   "sssp" : "sssp_delta_stepping_with_merge",
                                   "cc" : "cc_hybrid_dense_bitvec",
                                   "bfs" :"bfs_hybrid_dense",
                                   "bc" : "bc_SparsePushDensePull_bitvector",
                                   "tc" : "tc_hiroshi",
                                   "sssp_delta_stepping" : "sssp_delta_stepping_no_merge",
                                   "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
                       "twitter" : {"pr":"pagerank_pull",
                                    "sssp" : "sssp_delta_stepping_with_merge",
                                    "cc" : "cc_hybrid_dense_bitvec",
                                    "bfs" :"bfs_hybrid_dense_bitvec",
                                    "bc" : "bc_SparsePushDensePull_bitvector",
                                    "tc": "tc_hiroshi",
                                    "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                                    "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
                        "web" : {"pr":"pagerank_pull",
                                  "sssp" : "sssp_delta_stepping_with_merge",
                                  "cc" : "cc_hybrid_dense_bitvec",
                                  "bfs" :"bfs_hybrid_dense_bitvec",
                                  "bc" : "bc_SparsePushDensePull_bitvector",
                                  "tc": "tc_hiroshi",
                                  "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                                  "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
                        "kron" : {"pr":"pagerank_pull",
                                  "sssp" : "sssp_delta_stepping_with_merge",
                                  "cc" : "cc_hybrid_dense_bitvec",
                                  "bfs" :"bfs_hybrid_dense_bitvec",
                                  "bc" : "bc_SparsePushDensePull_bitvector",
                                  "tc": "tc_hiroshi",
                                  "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                                  "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
                        "urand" : {"pr":"pagerank_pull",
                                  "sssp" : "sssp_delta_stepping_with_merge",
                                  "cc" : "cc_hybrid_dense_bitvec",
                                  "bfs" :"bfs_hybrid_dense_bitvec",
                                  "bc" : "bc_SparsePushDensePull_bitvector",
                                  "tc": "tc_hiroshi",
                                  "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                                  "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},

                        "road" : {"pr":"pagerank_pull",
                                  "sssp" : "sssp_delta_stepping_with_merge",
                                  "cc" : "cc_hybrid_dense_bitvec",
                                  "bfs" :"bfs_hybrid_dense_bitvec",
                                  "bc" : "bc_SparsePushDensePull_bitvector",
                                  "tc": "tc_hiroshi",
                                  "sssp_delta_stepping" : "sssp_delta_stepping_with_merge",
                                  "sssp_delta_stepping_lazy" : "sssp_delta_stepping_lazy"},
                      }

graphit_PATH = "./bin/"
#DATA_PATH = "../../../data/"
DATA_PATH = "/data/sparse.tamu.edu/gap/"
OUTPUT_DIR= "./outputs/"

def get_vertex_count(graph):
    graph_to_count = {"twitter": 61578415, "web": 50636151, "kron": 134217726, 
                      "road": 23947347, "urand": 134217728}
 
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
    elif graph == "twitter":
        return ['12441072', '54488257', '25451915', '57714473', '14839494', '32081104', '52957357', '50444380', '49590701', '20127816', '34939333', '48251001', '19524253', '43676726', '33055508', '15244687', '24946738', '6479472', '26077682', '22023875', '22081915', '40034162', '49496014', '42847507', '52409557', '55445388', '22028097', '48766648', '44521241', '60135542', '28528671', '9678012', '40020306', '31625735', '37446892', '51788952', '52584255', '20346696', '48387909', '37337427', '50501084', '30130061', '41185893', '56495703', '45663305', '33359460', '48143058', '33291513', '53461445', '29340610', '34148498', '49171806', '35550696', '14521507', '51633218', '46823382', '19396273', '19871750', '36862677', '49539126', '34016452', '36567395', '55487793', '14391370']
    elif graph == "web":
        return ['10219452', '44758211', '890671', '13843756', '14168062', '20906930', '12189584', '26352335', '43500686', '8987024', '5699762', '41436455', '5030727', '40735218', '16533563', '28700166', '64711', '39634750', '16037779', '27152739', '16404061', '20491963', '5322423', '21420953', '26622109', '5882875', '18091040', '10665896', '18634422', '18138715', '2355535', '32885205', '40657440', '35196167', '45544426', '6175519', '40058318', '50626230', '36571019', '49397052', '23434265', '2299444', '32873823', '25978282', '2461715', '22787314', '30759947', '7428894', '39173870', '43194209', '26361509', '39747211', '30670029', '41483033', '9358666', '9945008', '3355244', '33831269', '45124744', '16137877', '11235448', '37509144', '27402414', '39546083']
    elif graph == "urand":
        return ['27691419', '121280314', '2413431', '37512113', '38390877', '56651037', '128461248', '33029842', '71406328', '117872827', '24351938', '15444519', '127526281', '112279428', '13631649', '110379302', '44800623', '77768193', '175347', '107397389', '43457209', '97215940', '73575165', '44449715', '33931724', '55526610', '14422051', '58043873', '72137329', '9647840', '15940695', '14209952', '49020883', '28901138', '50493273', '49150069', '126525082', '6382740', '89108297', '9239735', '110168548', '95370259', '116653530', '123410703', '16733665', '49030282', '108545121', '99095665', '133850077', '63499301', '21541382', '6230751', '89077456', '70392765', '6670455', '61746271', '83349535', '115272184', '20129908', '106148553', '117042375', '71431187', '45287808', '107702120']
    elif graph == "kron":
        return ['2338012', '31997659', '23590940', '43400604', '75337937', '169867', '104041220', '94177942', '32871357', '56230002', '69883037', '9346345', '48915358', '122571173', '6183279', '86323663', '106725780', '92389938', '16210738', '59816700', '111669929', '102831411', '113384800', '43872564', '80508827', '26105648', '8807516', '118452455', '121818859', '42361928', '29493053', '98461503', '71931337', '103808468', '4092345', '115276241', '4649343', '76656189', '31312001', '111334127', '100962918', '41823215', '22631240', '42848461', '79485148', '106818742', '73347974', '78848445', '109920510', '121492133', '101037296', '15438600', '4584784', '124503845', '87241743', '108297008', '33955082', '79934823', '8608481', '82435063', '46579271', '515421', '121530467', '127978736']
    elif graph == "road":
        return ['4795720', '21003853', '417968', '6496511', '6648699', '9811073', '22247478', '5720252', '12366459', '20413729', '4217374', '2674749', '22085557', '19445040', '2360788', '19115968', '7758767', '13468234', '30367', '18599547', '7526108', '16836280', '12742067', '7697995', '5876443', '9616340', '2497673', '10052290', '12493057', '1670855', '2760679', '2460941', '8489650', '5005225', '8744645', '8512023', '21912165', '1105390', '15432163', '1600177', '19079469', '16516637', '20202566', '21372803', '2898009', '8491277', '18798317', '23757560', '17161819', '23180739', '10997085', '3730630', '1079068', '15426822', '12190925', '1155218', '10693488', '14434835', '19963339', '3486185', '18383269', '20269908', '12370764', '7843140']
    else:
        print("Unsupported graph type")
        return []

def get_cmd_graphit(g, p, point):

    if p == "sssp" or p == "sssp_delta_stepping" or p == "sssp_delta_stepping_lazy":
        graph_path = DATA_PATH + g + ".wsg"

    #For Triangle Counting, we use undirected graph
    elif p == "tc" or p == "cc":
        graph_path = DATA_PATH + g + "U.sg"
    else:
        graph_path = DATA_PATH + g + ".sg"

    args = graph_path

    if p == "sssp" or p == "bfs" or p == "sssp_delta_stepping" or p == "sssp_delta_stepping_lazy":
        args += " " +  point
    if p == "sssp_delta_stepping" or p == "sssp_delta_stepping_lazy":
        args += " " + str(framework_app_graph_runtime_param_map["graphit"][p][g]) 
    
    
    command = graphit_PATH + graphit_binary_map[g][p] + " " + args
    #if p in ["pr", "cc", "prd"] and g in ["twitter", "webGraph", "friendster"]:
    
    if use_NUMACTL:
        # if NUAMCTL is available
        command = "numactl -i all " + command
    # TODO make it work for PR (try less than 16)
    command += " 16"

    return command


def get_cmd(framework, graph, app, point):
    if framework == "graphit":
        cmd = get_cmd_graphit(graph, app, point)
    return cmd

def get_bc_cmd(framework, graph, app, points):
    if framework == "graphit":
        cmd = get_bc_cmd_graphit(graph, app, points)
    return cmd

def get_bc_cmd_graphit(g, p, points):

    graph_path = DATA_PATH + g + ".sg"

    args = graph_path

    for point in points:
      args += " " + point
    command = graphit_PATH + graphit_binary_map[g][p] + " " + args
    if use_NUMACTL:
      command = "numactl -i all " + command
    return command




def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["road", "urand", "twitter", "web", "kron"], help = "enable graphs with socLive, road-usad, twitter, webGraph, friendster.Defaults to the test graph.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "pr", "cc", "tc", "bc", "ds"], 
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

                #process bc 
                if generic_app_name == "bc":
                  for i in range(0, len(points) - 3, 4):
                    starting_points = points[i:i+4]
                    cmd = get_bc_cmd(framework, graph, app, starting_points)
                    if not cmd:
                      break
                    print(cmd)
                    # setup timeout for executions that hang
                    kill = lambda process: process.kill()
                    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    timer = Timer(7200, kill, [out]) # timeout after 2 hrs
                    try:
                        timer.start()
                        (output, err) = out.communicate()
                        if err:
                          print("Failed with error", err)
                    finally:
                        timer.cancel()

                    if output:
                        successes += 1
                        log_file.write("\n------------------------------------------------\n")
                        log_file.write(time.strftime("%d/%m/%Y-%H:%M:%S"))
                        log_file.write("\n")
                        log_file.write(output)
                        log_file.write("\n------------------------------------------------\n")
                else:
                  for point in points:
                      cmd = get_cmd(framework, graph, app, point)
                      if not cmd:
                          break
                      print(cmd)

                      # setup timeout for executions that hang
                      kill = lambda process: process.kill()
                      out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                      timer = Timer(7200, kill, [out]) # timeout after 2 hrs
                      try:
                          timer.start()
                          (output, err) = out.communicate()
                          if err:
                            print("Failed with error", err)
                      finally:
                          timer.cancel()

                      if output:
                          successes += 1
                          log_file.write("\n------------------------------------------------\n")
                          log_file.write(time.strftime("%d/%m/%Y-%H:%M:%S"))
                          log_file.write("\n")
                          log_file.write(output.decode("utf-8"))
                          log_file.write("\n------------------------------------------------\n")
                          if generic_app_name in ["pr", "cc", "prd", "cf", "tc"]:
                              # pagerank, cc, prd, and cf can return when they succeeds once.
                              # greenmarl sets starting point internally
                              break;            
                log_file.write("successes=" + str(successes) + "\n")
                log_file.close()


if __name__ == "__main__":
    main()
