import os
import subprocess
import sys
DIR_PATH=os.path.dirname(os.path.realpath(__file__)).rstrip("/")

SCRATCH_PATH=""
GRAPHIT_BUILD_PATH=""
DATASET_PATH=""
APPS_DIRECTORY=""
GPU_ID=""
NVCC_PATH=""
CXX_COMPILER=""
NVCC_COMMAND=""
GPU_PREFIX=""


ORKUT=""
TWITTER=""
LIVEJOURNAL=""
SINAWEIBO=""
HOLLYWOOD=""
INDOCHINA=""
RUSA=""
RCA=""
RCENTRAL=""
GRAPH_ALL=[]
GRAPH_SOCIAL=[]
GRAPH_ROAD=[]

def find_dataset_files():
	global ORKUT
	global TWITTER
	global LIVEJOURNAL
	global SINAWEIBO
	global HOLLYWOOD
	global INDOCHINA
	global RUSA
	global RCA
	global RCENTRAL
	global GRAPH_ALL
	global GRAPH_ROAD
	global GRAPH_SOCIAL

	ORKUT=DATASET_PATH+"/soc-orkut.mtx"
	TWITTER=DATASET_PATH+"/soc-twitter-2010.mtx"
	LIVEJOURNAL=DATASET_PATH+"/soc-LiveJournal1.mtx"
	SINAWEIBO=DATASET_PATH+"/soc-sinaweibo.mtx"
	HOLLYWOOD=DATASET_PATH+"/hollywood-2009.weighted.mtx"
	INDOCHINA=DATASET_PATH+"/indochina-2004.weighted.mtx"
	RUSA=DATASET_PATH+"/road_usa.weighted.mtx"
	RCA=DATASET_PATH+"/roadNet-CA.weighted.mtx"
	RCENTRAL=DATASET_PATH+"/road_central.weighted.mtx"

	if len(sys.argv) >= 2 and sys.argv[1] == "small":	
		GRAPH_SOCIAL=[('livejournal', LIVEJOURNAL)]
		GRAPH_ROAD=[('rca', RCA)]
	else:
		GRAPH_SOCIAL=[('orkut', ORKUT), ('twitter', TWITTER), ('livejournal', LIVEJOURNAL), ('sinaweibo', SINAWEIBO), ('indochina', INDOCHINA), ('hollywood', HOLLYWOOD)]
		GRAPH_ROAD=[('rca', RCA), ('rusa', RUSA), ('rcentral', RCENTRAL)]

	GRAPH_ALL = GRAPH_SOCIAL + GRAPH_ROAD

	

def read_default_path(message, default):
	print(message + " [" + default + "]: ", end="")
	val = input().strip().rstrip("/")
	if val == "":
		val = default	
	return val

def get_gpu_count():
	gpus = os.popen("nvidia-smi -L").read().strip()
	return len(gpus.split("\n"))

def get_command_output(command):
	output = ""
	if isinstance(command, list):
		proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	else:
		proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	exitcode = proc.wait()
	if exitcode != 0:
		print(command)
	assert(exitcode == 0)
	for line in proc.stdout.readlines():
		if isinstance(line, bytes):
			line = line.decode()
		output += line.rstrip() + "\n"
	proc.stdout.close()
	return output

def set_NVCC_COMMAND(MAX_REG=64):
	global NVCC_COMMAND
	
	NVCC_COMMAND = NVCC_PATH + " -ccbin " + CXX_COMPILER + " "
	
	get_command_output(NVCC_COMMAND + APPS_DIRECTORY + "/obtain_gpu_cc.cu -o obtain_gpu_cc")
	output = get_command_output(GPU_PREFIX+"./obtain_gpu_cc").split()

	if len(output) != 2:
		print ("Cannot obtain GPU information")
		exit(-1)
	compute_capability = output[0]
	num_of_sm = output[1]

	if MAX_REG == 64:	
		NVCC_COMMAND += " -rdc=true -DNUM_CTA=" + str(int(num_of_sm)*2) + " -DCTA_SIZE=512 -gencode arch=compute_" + compute_capability + ",code=sm_" + compute_capability
	elif MAX_REG == 512:
		CTA_STYLE = (int(int(num_of_sm)/2), int(512/2))
		NVCC_COMMAND += " -rdc=true -DNUM_CTA=" + str(CTA_STYLE[0]) + " -DCTA_SIZE=" + str(CTA_STYLE[1]) + " -gencode arch=compute_" + compute_capability + ",code=sm_" + compute_capability
	else:
		print("Invalid MAX_REG configuration, not supported\n")
		exit(-1)

	NVCC_COMMAND += " -std=c++11 -O3 -I " + DIR_PATH+"/../.." + "/src/runtime_lib/ -Xcompiler \"-w\" -Wno-deprecated-gpu-targets --use_fast_math -Xptxas \" -dlcm=ca --maxrregcount=" + str(MAX_REG) + "\" "


def compile_application(gtfile, binname):
	if os.path.exists(binname):
		return
	get_command_output("python3 " + GRAPHIT_BUILD_PATH + "/bin/graphitc.py -f " + APPS_DIRECTORY + "/" + gtfile + " -o " + gtfile + ".cu")
	get_command_output(NVCC_COMMAND + gtfile + ".cu -o " + binname)


def run_sanity_check():
	compile_application("simple_graph_load.gt", "load")
	get_command_output(GPU_PREFIX+"./load " + RCA)


def compile_and_run(gtfile, binname, run_args, outputf):
	compile_application(gtfile, binname)
	output = get_command_output(GPU_PREFIX+"./"+binname + " " + run_args)
	f = open(outputf, "w")
	f.write(output)
	f.close()


def run_pr():
	set_NVCC_COMMAND()
	print("Running eval for Pagerank")
	PR = "pr.gt"	
	for i, (name, graph) in enumerate(GRAPH_ALL):
		compile_and_run(PR, "pr", graph, "pr_" + name + ".out")
		print(str(i+1) + "/" + str(len(GRAPH_ALL)))


def run_cc():
	set_NVCC_COMMAND()
	print("Running eval for Connected Components")
	CC = "cc.gt"	
	for i, (name, graph) in enumerate(GRAPH_ALL):
		compile_and_run(CC, "cc", graph, "cc_" + name + ".out")
		print(str(i+1) + "/" + str(len(GRAPH_ALL)))


def run_ds():
	delta = {}
	delta["orkut"] = 22
	delta["livejournal"] = 120
	delta["twitter"] = 15
	delta["sinaweibo"] = 15
	delta["hollywood"] = 15
	delta["indochina"] = 10000
	delta["rusa"] = 80000
	delta["rcentral"] = 30000
	delta["rca"] = 20000
	
	print ("Running eval for Delta Stepping")
	DS_SOCIAL = "ds_social.gt"
	DS_ROAD = "ds_road.gt"
	set_NVCC_COMMAND()
	for i, (name, graph) in enumerate(GRAPH_SOCIAL):
		compile_and_run(DS_SOCIAL, "ds_social", graph + " 0 " + str(delta[name]), "ds_" + name + ".out")
		print(str(i+1) + "/" + str(len(GRAPH_ALL)))

	set_NVCC_COMMAND(512)
	for i, (name, graph) in enumerate(GRAPH_ROAD):
		compile_and_run(DS_ROAD, "ds_road", graph + " 0 " + str(delta[name]), "ds_" + name + ".out")
		print(str(i+1+len(GRAPH_SOCIAL)) + "/" + str(len(GRAPH_ALL)))
		
def run_bc():
	threshold = {}
	threshold["orkut"] = 0.010
	threshold["livejournal"] = 0.006
	threshold["twitter"] = 0.023
	threshold["sinaweibo"] = 0.008
	threshold["hollywood"] = 0.026
	threshold["indochina"] = 0.99
	
	print ("Running eval for Betweenness Centrality")	
	BC_SOCIAL = "bc_social.gt"
	BC_ROAD = "bc_road.gt"
	set_NVCC_COMMAND()
	for i, (name, graph) in enumerate(GRAPH_SOCIAL):
		compile_and_run(BC_SOCIAL, "bc_social", graph + " 0 " + str(threshold[name]), "bc_" + name + ".out")
		print(str(i+1) + "/" + str(len(GRAPH_ALL)))
	set_NVCC_COMMAND(512)
	for i, (name, graph) in enumerate(GRAPH_ROAD):
		compile_and_run(BC_ROAD, "bc_road", graph + " 0", "bc_" + name + ".out")
		print(str(i+1+len(GRAPH_SOCIAL)) + "/" + str(len(GRAPH_ALL)))
	

def run_bfs():
	threshold = {}
	threshold["orkut"] = 0.010
	threshold["livejournal"] = 0.006
	threshold["twitter"] = 0.023
	threshold["sinaweibo"] = 0.008
	threshold["hollywood"] = 0.026
	threshold["indochina"] = 0.99
	
	print ("Running eval for Breadth First Search")	
	BFS_SOCIAL = "bfs_social.gt"
	BFS_ROAD = "bfs_road.gt"
	set_NVCC_COMMAND()
	for i, (name, graph) in enumerate(GRAPH_SOCIAL):
		compile_and_run(BFS_SOCIAL, "bfs_social", graph + " 0 " + str(threshold[name]), "bfs_" + name + ".out")
		print(str(i+1) + "/" + str(len(GRAPH_ALL)))
	set_NVCC_COMMAND(512)
	for i, (name, graph) in enumerate(GRAPH_ROAD):
		compile_and_run(BFS_ROAD, "bfs_road", graph + " 0", "bfs_" + name + ".out")
		print(str(i+1+len(GRAPH_SOCIAL)) + "/" + str(len(GRAPH_ALL)))


def read_execution_time(filename):
	try:
		f = open(SCRATCH_PATH + "/" + filename, "r")	
		values = f.read().strip().split("\n")
		values = [float(val) for val in values]
		min_val = min(values)
		min_val = int(min_val * 100000) / 100.0
		return min_val
	except:
		return -1
   

def run_tests():
	# get the GPU properties first
	set_NVCC_COMMAND()
	run_sanity_check()
	run_pr()
	run_cc()
	run_ds()
	run_bc()
	run_bfs()


def print_cell(f, val):
	spaces = 9 - len(str(val))
	f.write(" " * spaces + str(val) + " |")

def gen_table7():
	short_names = {}
	short_names["orkut"] = "OK"
	short_names["twitter"] = "TW"
	short_names["livejournal"] = "LJ"
	short_names["sinaweibo"] = "SW"
	short_names["hollywood"] = "HW"
	short_names["indochina"] = "IC"
	short_names["rusa"] = "RU"
	short_names["rca"] = "RN"
	short_names["rcentral"] = "RC"

	filepath = SCRATCH_PATH + "/table7.txt"
	f = open(filepath, "w")
	
	f.write("-" * 67)
	f.write("\n")
	f.write("|")
	print_cell(f, "Graph")
	print_cell(f, "PR")
	print_cell(f, "CC")
	print_cell(f, "BFS")
	print_cell(f, "BC")
	print_cell(f, "SSSP")
	f.write("\n")
	f.write("-" * 67)
	f.write("\n")
	
	for graph, _  in GRAPH_ALL:
		f.write("|")
		print_cell(f, short_names[graph])
		for app in ["pr", "cc", "bfs", "bc", "ds"]:
			fname = app + "_" + graph + ".out"
			val = read_execution_time(fname)
			print_cell(f, val)
		f.write("\n")
	
	f.write("-" * 67)
	f.write("\n")
	
	f.close()
	print(open(filepath, "r").read())
	print("# This table is generated at: ", filepath)

	
def main():
	global SCRATCH_PATH
	global GRAPHIT_BUILD_PATH
	global DATASET_PATH
	global APPS_DIRECTORY
	global GPU_ID
	global NVCC_PATH
	global CXX_COMPILER
	global GPU_PREFIX

	print("Starting artifact evaluation in directory: ", DIR_PATH)
	SCRATCH_PATH = read_default_path("Please choose a output directory to use", DIR_PATH + "/table7_outputs")
	GRAPHIT_BUILD_PATH = read_default_path("Please choose GraphIt build directory", DIR_PATH + "/../../build")
	DATASET_PATH = read_default_path("Please choose dataset path", DIR_PATH + "/dataset")
	APPS_DIRECTORY = DIR_PATH+"/table7_inputs"
	NVCC_PATH = read_default_path("Please choose NVCC path", "/usr/local/cuda/bin/nvcc")
	CXX_COMPILER = read_default_path("Please choose CXX_COMPILER", "/usr/bin/g++")

	if os.path.exists(SCRATCH_PATH):
		os.system("rm -rf " + SCRATCH_PATH)
	os.makedirs(SCRATCH_PATH)

	os.chdir(SCRATCH_PATH)


	total_devices = get_gpu_count()
	GPU_ID = read_default_path("Choose GPU id to use (0-" + str(total_devices-1) + ")", str(0))
	GPU_PREFIX="CUDA_VISIBLE_DEVICES="+GPU_ID+" "
	

	find_dataset_files()
	run_tests()
	gen_table7()






if __name__ == "__main__":
	main()
