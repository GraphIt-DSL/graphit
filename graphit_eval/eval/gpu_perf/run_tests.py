import os
import sys
import subprocess
import shutil

NVCC_PATH="/usr/local/cuda/bin/nvcc"

GRAPHIT_SRC_DIR=""
GRAPHIT_BUILD_DIR=""
GRAPH_DIR=""

WORKING_DIR=os.path.abspath("./scratch").rstrip("/")

OUTPUT_DIR=os.path.abspath("./output").rstrip("/")
INPUTS_DIR=os.path.abspath("./inputs").rstrip("/")


GPU_CC=""
NUM_SM=""

def get_command_output_class(command):
	output = ""
	if isinstance(command, list):
		proc = subprocess.Popen(command, stdout=subprocess.PIPE)
	else:
		print(command)
		proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	exitcode = proc.wait()
	for line in proc.stdout.readlines():
		if isinstance(line, bytes):
			line = line.decode()
		output += line.rstrip() + "\n"

	proc.stdout.close()
	return exitcode, output

def get_command_output(command):
	(exitcode, output) = get_command_output_class(command)
	if exitcode != 0:
		print("Error executing command:", command)
		exit(1)
	return output

def get_gpu_prop():
	global GPU_CC
	global NUM_SM
	global NVCC_PATH
	global GRAPHIT_SRC_DIR
	global WORKING_DIR

	get_command_output(NVCC_PATH + " " + GRAPHIT_SRC_DIR + "/test/gpu_tests/test_input/obtain_gpu_cc.cu -o " + WORKING_DIR + "/obtain_gpu_cc")
	output = get_command_output(WORKING_DIR+"/obtain_gpu_cc").strip().split("\n")
	if len(output) != 2:
		print("Cannot obtain GPU information")
		exit(1)
	GPU_CC=output[0]
	NUM_SM=output[1]	


def compile_and_execute(input_file, graph_name, args, output_name):
	global GRAPHIT_SRC_DIR
	global GRAPHIT_BUILD_DIR
	global GRAPH_DIR
	global WORKING_DIR
	global OUTPUT_DIR
	global GPU_CC
	global NUM_SM
	global NVCC_PATH
	
	nvcc_command = NVCC_PATH + " -rdc=true --use_fast_math -Xptxas \"-dlcm=ca --maxrregcount=64\" -std=c++11 -DNUM_CTA=" + str(int(NUM_SM)*2)+ " -DCTA_SIZE=512 -gencode arch=compute_" + GPU_CC + ",code=sm_"+GPU_CC	

	graphit_compiler_command = "python " + GRAPHIT_BUILD_DIR + "/bin/graphitc.py -o " + WORKING_DIR+"/test_cpp.cu -f"


	cwd = os.getcwd()
	os.chdir(WORKING_DIR)
	get_command_output(graphit_compiler_command + " " + input_file)
	get_command_output(nvcc_command + " " + WORKING_DIR+"/test_cpp.cu -o " + WORKING_DIR+"/test_executable -I " + GRAPHIT_SRC_DIR+"/src/runtime_lib")
	output = get_command_output(WORKING_DIR+"/test_executable " + graph_name + " " + args)
	os.chdir(cwd)
	
	f = open(OUTPUT_DIR+"/"+output_name, "w")
	f.write(output)
	f.close()
	

def parse_output_file(output_name):
	global OUTPUT_DIR
	f = open(OUTPUT_DIR+"/"+output_name)
	content = f.read().strip().split("\n")
	f.close()
	min_time = 1000000
	for line in content:
		try:
			time = float(line)
		except ValueError as verr:
			time = -1
		if time == -1:
			continue
		if time < min_time:
			min_time = time
	return time

def create_csv(time_values, output_name):
	global OUTPUT_DIR
	f = open(OUTPUT_DIR+"/"+output_name, "w")
	
	for graph in time_values.keys():
		f.write (graph+", " + str(time_values[graph]) + "\n")

	f.close()

def test_pr():
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/soc-orkut.mtx", "", "pr_OK")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/soc-twitter-2010.mtx", "", "pr_TW")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/soc-LiveJournal1.mtx", "", "pr_LJ")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/soc-sinaweibo.mtx", "", "pr_SW")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/indochina-2004.weighted.mtx", "", "pr_IC")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/hollywood-2009.weighted.mtx", "", "pr_HW")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/road_central.weighted.mtx", "", "pr_RC")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/road_usa.weighted.mtx", "", "pr_RU")
	compile_and_execute(INPUTS_DIR+"/pr_social.gt", GRAPH_DIR+"/roadNet-CA.weighted.mtx", "", "pr_RN")

	time_values={}
	time_values['OK'] = parse_output_file("pr_OK")
	time_values['TW'] = parse_output_file("pr_TW")
	time_values['LJ'] = parse_output_file("pr_LJ")
	time_values['SW'] = parse_output_file("pr_SW")
	time_values['IC'] = parse_output_file("pr_IC")
	time_values['HW'] = parse_output_file("pr_HW")
	time_values['RC'] = parse_output_file("pr_RC")
	time_values['RU'] = parse_output_file("pr_RU")
	time_values['RN'] = parse_output_file("pr_RN")

	create_csv(time_values, "pr.csv")

def test_sssp():
	compile_and_execute(INPUTS_DIR+"/sssp_power.gt", GRAPH_DIR+"/soc-orkut.mtx", "0 22", "sssp_OK")
	compile_and_execute(INPUTS_DIR+"/sssp_power.gt", GRAPH_DIR+"/soc-twitter-2010.mtx", "0 15", "sssp_TW")
	compile_and_execute(INPUTS_DIR+"/sssp_power.gt", GRAPH_DIR+"/soc-LiveJournal1.mtx", "0 120", "sssp_LJ")
	compile_and_execute(INPUTS_DIR+"/sssp_power.gt", GRAPH_DIR+"/soc-sinaweibo.mtx", "0 15", "sssp_SW")
	compile_and_execute(INPUTS_DIR+"/sssp_power.gt", GRAPH_DIR+"/indochina-2004.weighted.mtx", "0 15", "sssp_IC")
	compile_and_execute(INPUTS_DIR+"/sssp_power.gt", GRAPH_DIR+"/hollywood-2009.weighted.mtx", "0 15", "sssp_HW")
	compile_and_execute(INPUTS_DIR+"/sssp_road.gt", GRAPH_DIR+"/road_central.weighted.mtx", "0 80000", "sssp_RC")
	compile_and_execute(INPUTS_DIR+"/sssp_road.gt", GRAPH_DIR+"/road_usa.weighted.mtx", "0 80000", "sssp_RU")
	compile_and_execute(INPUTS_DIR+"/sssp_road.gt", GRAPH_DIR+"/roadNet-CA.weighted.mtx", "0 80000", "sssp_RN")

	time_values={}
	time_values['OK'] = parse_output_file("sssp_OK")
	time_values['TW'] = parse_output_file("sssp_TW")
	time_values['LJ'] = parse_output_file("sssp_LJ")
	time_values['SW'] = parse_output_file("sssp_SW")
	time_values['IC'] = parse_output_file("sssp_IC")
	time_values['HW'] = parse_output_file("sssp_HW")
	time_values['RC'] = parse_output_file("sssp_RC")
	time_values['RU'] = parse_output_file("sssp_RU")
	time_values['RN'] = parse_output_file("sssp_RN")

	create_csv(time_values, "sssp.csv")

def test_cc():
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/soc-orkut.mtx", "", "cc_OK")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/soc-twitter-2010.mtx", "", "cc_TW")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/soc-LiveJournal1.mtx", "", "cc_LJ")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/soc-sinaweibo.mtx", "", "cc_SW")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/indochina-2004.weighted.mtx", "", "cc_IC")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/hollywood-2009.weighted.mtx", "", "cc_HW")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/road_central.weighted.mtx", "", "cc_RC")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/road_usa.weighted.mtx", "", "cc_RU")
	compile_and_execute(INPUTS_DIR+"/cc_power.gt", GRAPH_DIR+"/roadNet-CA.weighted.mtx", "", "cc_RN")

	time_values={}
	time_values['OK'] = parse_output_file("cc_OK")
	time_values['TW'] = parse_output_file("cc_TW")
	time_values['LJ'] = parse_output_file("cc_LJ")
	time_values['SW'] = parse_output_file("cc_SW")
	time_values['IC'] = parse_output_file("cc_IC")
	time_values['HW'] = parse_output_file("cc_HW")
	time_values['RC'] = parse_output_file("cc_RC")
	time_values['RU'] = parse_output_file("cc_RU")
	time_values['RN'] = parse_output_file("cc_RN")

	create_csv(time_values, "cc.csv")

def test_bfs():
	compile_and_execute(INPUTS_DIR+"/bfs_power.gt", GRAPH_DIR+"/soc-orkut.mtx", "0 0.12", "bfs_OK")
	compile_and_execute(INPUTS_DIR+"/bfs_power.gt", GRAPH_DIR+"/soc-twitter-2010.mtx", "0 0.03", "bfs_TW")
	compile_and_execute(INPUTS_DIR+"/bfs_power.gt", GRAPH_DIR+"/soc-LiveJournal1.mtx", "0 0.015", "bfs_LJ")
	compile_and_execute(INPUTS_DIR+"/bfs_power.gt", GRAPH_DIR+"/soc-sinaweibo.mtx", "0 0.012", "bfs_SW")
	compile_and_execute(INPUTS_DIR+"/bfs_power.gt", GRAPH_DIR+"/indochina-2004.weighted.mtx", "0 0.03", "bfs_IC")
	compile_and_execute(INPUTS_DIR+"/bfs_power.gt", GRAPH_DIR+"/hollywood-2009.weighted.mtx", "0 0.03", "bfs_HW")
	compile_and_execute(INPUTS_DIR+"/bfs_road.gt", GRAPH_DIR+"/road_central.weighted.mtx", "0", "bfs_RC")
	compile_and_execute(INPUTS_DIR+"/bfs_road.gt", GRAPH_DIR+"/road_usa.weighted.mtx", "0", "bfs_RU")
	compile_and_execute(INPUTS_DIR+"/bfs_road.gt", GRAPH_DIR+"/roadNet-CA.weighted.mtx", "0", "bfs_RN")

	time_values={}
	time_values['OK'] = parse_output_file("bfs_OK")
	time_values['TW'] = parse_output_file("bfs_TW")
	time_values['LJ'] = parse_output_file("bfs_LJ")
	time_values['SW'] = parse_output_file("bfs_SW")
	time_values['IC'] = parse_output_file("bfs_IC")
	time_values['HW'] = parse_output_file("bfs_HW")
	time_values['RC'] = parse_output_file("bfs_RC")
	time_values['RU'] = parse_output_file("bfs_RU")
	time_values['RN'] = parse_output_file("bfs_RN")

	create_csv(time_values, "bfs.csv")

def test_bc():
	compile_and_execute(INPUTS_DIR+"/bc_power.gt", GRAPH_DIR+"/soc-orkut.mtx", "0 0.12", "bc_OK")
	compile_and_execute(INPUTS_DIR+"/bc_power.gt", GRAPH_DIR+"/soc-twitter-2010.mtx", "0 0.03", "bc_TW")
	compile_and_execute(INPUTS_DIR+"/bc_power.gt", GRAPH_DIR+"/soc-LiveJournal1.mtx", "0 0.015", "bc_LJ")
	compile_and_execute(INPUTS_DIR+"/bc_power.gt", GRAPH_DIR+"/soc-sinaweibo.mtx", "0 0.012", "bc_SW")
	compile_and_execute(INPUTS_DIR+"/bc_power.gt", GRAPH_DIR+"/indochina-2004.weighted.mtx", "0 0.03", "bc_IC")
	compile_and_execute(INPUTS_DIR+"/bc_power.gt", GRAPH_DIR+"/hollywood-2009.weighted.mtx", "0 0.03", "bc_HW")
	compile_and_execute(INPUTS_DIR+"/bc_road.gt", GRAPH_DIR+"/road_central.weighted.mtx", "0", "bc_RC")
	compile_and_execute(INPUTS_DIR+"/bc_road.gt", GRAPH_DIR+"/road_usa.weighted.mtx", "0", "bc_RU")
	compile_and_execute(INPUTS_DIR+"/bc_road.gt", GRAPH_DIR+"/roadNet-CA.weighted.mtx", "0", "bc_RN")

	time_values={}
	time_values['OK'] = parse_output_file("bc_OK")
	time_values['TW'] = parse_output_file("bc_TW")
	time_values['LJ'] = parse_output_file("bc_LJ")
	time_values['SW'] = parse_output_file("bc_SW")
	time_values['IC'] = parse_output_file("bc_IC")
	time_values['HW'] = parse_output_file("bc_HW")
	time_values['RC'] = parse_output_file("bc_RC")
	time_values['RU'] = parse_output_file("bc_RU")
	time_values['RN'] = parse_output_file("bc_RN")

	create_csv(time_values, "bc.csv")

def run_all_tests():
	test_pr()	
	test_sssp()	
	test_cc()	
	test_bfs()	
	test_bc()	

def usage(pname):
	print("Usage:")
	print(pname + " <graphit_src_dir> <graphit_build_dir> <graph_directory_path>")

def main():
	global GRAPHIT_SRC_DIR
	global GRAPHIT_BUILD_DIR
	global GRAPH_DIR
	global WORKING_DIR
	global OUTPUT_DIR

	if len(sys.argv) < 4:
		usage(sys.argv[0])
		exit(1)
	GRAPHIT_SRC_DIR = os.path.abspath(sys.argv[1].strip()).rstrip("/")
	GRAPHIT_BUILD_DIR = os.path.abspath(sys.argv[2].strip()).rstrip("/")
	GRAPH_DIR = os.path.abspath(sys.argv[3].strip()).rstrip("/")


	if os.path.isdir(WORKING_DIR):
		shutil.rmtree(WORKING_DIR)
	os.mkdir(WORKING_DIR)


	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

	get_gpu_prop()	

	run_all_tests()

if __name__ == '__main__':
	main()
	
