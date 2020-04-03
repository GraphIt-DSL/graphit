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
#    (2) set the applications, all of the applications are following graphit apps (mapped into other framewroks)

LOG_PATH = "./outputs/"

def get_starting_points(graph):
    """ Use the points with non-zero out degree and don't hang during execution.  """
    if graph == "testGraph":
        return ["1","2"]
    elif graph == "twitter":
        return ['12441073', '54488258', '25451916', '57714474', '14839495', '32081105', '52957358', '50444381', '49590702', '20127817', '34939334', '48251002', '19524254', '43676727', '33055509', '15244688', '24946739', '6479473', '26077683', '22023876', '22081916', '40034163', '49496015', '42847508', '52409558', '55445389', '22028098', '48766649', '44521242', '60135543', '28528672', '9678013', '40020307', '31625736', '37446893', '51788953', '52584256', '20346697', '48387910', '37337428', '50501085', '30130062', '41185894', '56495704', '45663306', '33359461', '48143059', '33291514', '53461446', '29340611', '34148499', '49171807', '35550697', '14521508', '51633219', '46823383', '19396274', '19871751', '36862678', '49539127', '34016453', '36567396', '55487794', '14391371']
    elif graph == "web":
        return ['10219453', '44758212', '890672', '13843757', '14168063', '20906931', '12189585', '26352336', '43500687', '8987025', '5699763', '41436456', '5030728', '40735219', '16533564', '28700167', '64712', '39634751', '16037780', '27152740', '16404062', '20491964', '5322424', '21420954', '26622110', '5882876', '18091041', '10665897', '18634423', '18138716', '2355536', '32885206', '40657441', '35196168', '45544427', '6175520', '40058319', '50626231', '36571020', '49397053', '23434266', '2299445', '32873824', '25978283', '2461716', '22787315', '30759948', '7428895', '39173871', '43194210', '26361510', '39747212', '30670030', '41483034', '9358667', '9945009', '3355245', '33831270', '45124745', '16137878', '11235449', '37509145', '27402415', '39546084'] 
    elif graph == "urand":
        return ['27691420', '121280315', '2413432', '37512114', '38390878', '56651038', '128461249', '33029843', '71406329', '117872828', '24351939', '15444520', '127526282', '112279429', '13631650', '110379303', '44800624', '77768194', '175348', '107397390', '43457210', '97215941', '73575166', '44449716', '33931725', '55526611', '14422052', '58043874', '72137330', '9647841', '15940696', '14209953', '49020884', '28901139', '50493274', '49150070', '126525083', '6382741', '89108298', '9239736', '110168549', '95370260', '116653531', '123410704', '16733666', '49030283', '108545122', '99095666', '133850078', '63499302', '21541383', '6230752', '89077457', '70392766', '6670456', '61746272', '83349536', '115272185', '20129909', '106148554', '117042376', '71431188', '45287809', '107702121']
    elif graph == "kron":
        return ['2338013', '31997660', '23590941', '43400605', '75337938', '169868', '104041221', '94177943', '32871358', '56230003', '69883038', '9346346', '48915359', '122571174', '6183280', '86323664', '106725781', '92389939', '16210739', '59816701', '111669930', '102831412', '113384801', '43872565', '80508828', '26105649', '8807517', '118452456', '121818860', '42361929', '29493054', '98461504', '71931338', '103808469', '4092346', '115276242', '4649344', '76656190', '31312002', '111334128', '100962919', '41823216', '22631241', '42848462', '79485149', '106818743', '73347975', '78848446', '109920511', '121492134', '101037297', '15438601', '4584785', '124503846', '87241744', '108297009', '33955083', '79934824', '8608482', '82435064', '46579272', '515422', '121530468', '127978737']
    elif graph == "road":
        return ['4795721', '21003854', '417969', '6496512', '6648700', '9811074', '22247479', '5720253', '12366460', '20413730', '4217375', '2674750', '22085558', '19445041', '2360789', '19115969', '7758768', '13468235', '30368', '18599548', '7526109', '16836281', '12742068', '7697996', '5876444', '9616341', '2497674', '10052291', '12493058', '1670856', '2760680', '2460942', '8489651', '5005226', '8744646', '8512024', '21912166', '1105391', '15432164', '1600178', '19079470', '16516638', '20202567', '21372804', '2898010', '8491278', '18798318', '23757561', '17161820', '23180740', '10997086', '3730631', '1079069', '15426823', '12190926', '1155219', '10693489', '14434836', '19963340', '3486186', '18383270', '20269909', '12370765', '7843141']
    else:
        print "Unsupported graph type"
        return []


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
    print ("processing log file: " + log_file_name)
    with open(log_file_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    # if the file is empty, don't try to parse it
    if (len(content) < 3):
        print "invalid log file" + log_file_name
        return -1
    initial_inner_cnt = inner_cnt
    min_time = 10000000
    sum_time = 0
    successes = 1
    for i in range(len(content)):
        line = content[i]
        if line.find("successes") != -1:
            successes = int(line.split("=")[1])
        if line.find(time_key) != -1:
            if time_key_own_line:
                next_line = content[i+1]
                time_str = next_line.strip()
            else:
                time_str = line.split(delimiter)[index]
            if strip_end:
                time_str = time_str[:strip_end] # strip the (s) at the end
            if time_str.strip() == '':
                time = min_time
            else:
                time = float(time_str) / divider 
            if time < min_time:
                min_time = time
            inner_cnt -= 1
            if inner_cnt == 0 and (app == "bfs" or app == "sssp"):
                sum_time += min_time
                min_time = 10000
                inner_cnt = initial_inner_cnt
    if (app == "pr" or app == "cc" or app == "cf" or app == "prd"):
        return min_time
    else: # bfs or sssp
        return sum_time / successes

def process_bfs(log_file_name, app, time_key, delimiter, index, strip_end, divider, inner_cnt, time_key_own_line):
    """
    @time_key: used to find the time of execution in the log file
    @delimiter: used to extract the time
    @index: used together with the delimiter
    @strip_end: end index to strip off (e.g. seconds) from the time value
    @divider: divice this value to convert the time in log into seconds
    @inner_cnt: number of runs that an application performs internally
    @time_key_own_line: if the time_key is one its own line
    """
    print ("processing log file: " + log_file_name)
    with open(log_file_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    # if the file is empty, don't try to parse it
    if (len(content) < 3):
        print "invalid log file" + log_file_name
        return -1
    initial_inner_cnt = inner_cnt
    min_time = 10000000
    sum_time = 0
    successes = 0
    for i in range(len(content)):
        line = content[i]
        if line.find(time_key) != -1:
            if time_key_own_line:
                next_line = content[i+1]
                time_str = next_line.strip()
            else:
                time_str = line.split(delimiter)[index]
            if strip_end:
                time_str = time_str[:strip_end] # strip the (s) at the end
            if time_str.strip() == '':
                time = min_time
            else:
                time = float(time_str) / divider 
            sum_time += time
            successes += 1
    return sum_time / successes

def process_tc(log_file_name, app, time_key, delimiter, index, strip_end, divider, inner_cnt, time_key_own_line):
    """
    @time_key: used to find the time of execution in the log file
    @delimiter: used to extract the time
    @index: used together with the delimiter
    @strip_end: end index to strip off (e.g. seconds) from the time value
    @divider: divice this value to convert the time in log into seconds
    @inner_cnt: number of runs that an application performs internally
    @time_key_own_line: if the time_key is one its own line
    """
    print ("processing log file: " + log_file_name)
    with open(log_file_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    # if the file is empty, don't try to parse it
    if (len(content) < 3):
        print "invalid log file" + log_file_name
        return -1
    initial_inner_cnt = inner_cnt
    min_time = 10000000
    sum_time = 0
    successes = 0
    for i in range(len(content)):
        line = content[i]
        if line.find(time_key) != -1:
            if time_key_own_line:
                next_line = content[i+1]
                time_str = next_line.strip()
            else:
                time_str = line.split(delimiter)[index]
            if strip_end:
                time_str = time_str[:strip_end] # strip the (s) at the end
            if time_str.strip() == '':
                time = min_time
            else:
                time = float(time_str) / divider 
            sum_time += time
            successes += 1

    return sum_time / successes



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

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--frameworks', nargs='+',
                        default=["graphit"], # "netflix" only used for cf
                        help="frameworks to parse")
    parser.add_argument('-g', '--graphs', nargs='+',
                        default=["road", "urand", "twitter", "web", "kron"], help = "enable graphs with socLive, road-usad, twitter, webGraph, friendster.Defaults to the test graph.")
    parser.add_argument('-a', '--applications', nargs='+',
                        default=["bfs", "pr", "cc", "tc", "bc", "ds"], 
                        help="applications to benchmark. Defaults to all four applications.")
    args = parser.parse_args()

    # time_key, delimiter, index, strip_end, divider, inner_cnt, time_key_own_line
    parse_args = {"greenmarl": ["running time", "=", 1, 0, 1000, 1, False], # runs 10 times internally so 1000*10
                  "graphit": ["elapsed time", "", None, None, 1, 10, True],
                  "ligra": ["Running time", ":", 1, 0, 1, 3, False],
                  "galois": [",Time,", ",", 4, 0, 1000, 1, False],
                  "gemini": ["exec_time", "=", 1, -3, 1, 6, False],
                  "grazelle": ["Running Time", "= ", 1, -2, 1000, 1, False],
                  "polymer": ["", "= ", 1, -2, 1000, 1, False]}

    results = {}
    for framework in args.frameworks:
        results[framework] = {}
        for g in args.graphs:
            print("graph: " + g)
            results[framework][g] = {}
            for app in args.applications:
                log_file_name = LOG_PATH + framework + "/" + app + "_" + g + ".txt"
                if app == "bc":
                    results[framework][g][app] = process_tc(log_file_name, app, *parse_args[framework])
                elif app == "tc":
                    results[framework][g][app] = process_tc(log_file_name, app, *parse_args[framework])
                elif app == "bfs":
                    results[framework][g][app] = process_bfs(log_file_name, app, *parse_args[framework])
                elif app == "cc":
                    results[framework][g][app] = process_tc(log_file_name, app, *parse_args[framework])
                elif app == "ds":
                    results[framework][g][app] = process_bfs(log_file_name, app, *parse_args[framework])
                else:
                    runtime = parse_result(log_file_name, app, *parse_args[framework])
                    if framework == "greenmarl" and app in ["bfs", "sssp"]:
                        # greenmarl internally picks 10 starting points for sssp and bfs
                        # but there is only 1 successes
                        runtime /= 10
                    results[framework][g][app] = runtime
    print results
    #print_normalized_execution(args.frameworks, args.applications, args.graphs, results)
    print_absolute_execution(args.frameworks, args.applications, args.graphs, results)

    
if __name__ == "__main__":
    main()

