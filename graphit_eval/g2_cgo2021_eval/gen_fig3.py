import os
import subprocess
DIR_PATH=os.path.dirname(os.path.realpath(__file__)).rstrip("/")

SCRATCH_PATH=""
GRAPHIT_BUILD_PATH=""
APPS_DIRECTORY=""

	

def read_default_path(message, default):
	print(message + " [" + default + "]: ", end="")
	val = input().strip().rstrip("/")
	if val == "":
		val = default	
	return val

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

def compile_application(gtfile):
	get_command_output("python3 " + GRAPHIT_BUILD_PATH + "/bin/graphitc.py -f " + APPS_DIRECTORY + "/" + gtfile + " -o " + gtfile + ".cu")
	

def run_tests():
	compile_application("fig3_a.gt")
	compile_application("fig3_b.gt")
	compile_application("fig3_c.gt")
	
	os.system("rm compile.cpp compile.o")
	
	
def main():
	global SCRATCH_PATH
	global GRAPHIT_BUILD_PATH
	global APPS_DIRECTORY

	print("Starting artifact evaluation in directory: ", DIR_PATH)
	SCRATCH_PATH = read_default_path("Please choose a output directory to use", DIR_PATH + "/fig3_outputs")
	GRAPHIT_BUILD_PATH = read_default_path("Please choose GraphIt build directory", DIR_PATH + "/../../build")
	APPS_DIRECTORY = DIR_PATH+"/fig3_inputs"
	
	if os.path.exists(SCRATCH_PATH):
		os.system("rm -rf " + SCRATCH_PATH)
	os.makedirs(SCRATCH_PATH)
	
	os.chdir(SCRATCH_PATH)
	

	run_tests()






if __name__ == "__main__":
	main()
