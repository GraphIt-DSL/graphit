import os
import tempfile
import subprocess
import importlib.util
import platform
import pybind11

GRAPHIT_BUILD_DIRECTORY="${GRAPHIT_BUILD_DIRECTORY}".strip().rstrip("/")
GRAPHIT_SOURCE_DIRECTORY="${GRAPHIT_SOURCE_DIRECTORY}".strip().rstrip("/")
CXX_COMPILER="${CXX_COMPILER}".strip().rstrip("/")

module_so_list = []
def compile_and_load(graphit_source_file):
	# Obtain a unique filename for the module
	module_file = tempfile.NamedTemporaryFile()
	module_filename_base = module_file.name
	module_file.close()

	# compile the file into a cpp file
	module_filename_cpp = module_filename_base + ".cpp"
	module_filename_object = module_filename_base + ".o"
	module_filename_so = module_filename_base + ".so"
	module_name = os.path.basename(module_filename_base)	
	
	subprocess.check_call("python " + GRAPHIT_BUILD_DIRECTORY + "/bin/graphitc.py -f " + graphit_source_file + " -o " + module_filename_cpp + " -m " + module_name, shell=True)
	
	# now compile the file into .so
	subprocess.check_call(CXX_COMPILER + " -I" + pybind11.get_include() + " $(python3-config --includes) -c " + module_filename_cpp + " -I " + GRAPHIT_SOURCE_DIRECTORY + "/src/runtime_lib/ -std=c++11 -DGEN_PYBIND_WRAPPERS -flto -fno-fat-lto-objects -fPIC -fvisibility=hidden -o " + module_filename_object, shell=True)

	cmd = CXX_COMPILER + " -fPIC -shared -o " + module_filename_so + " " + module_filename_object + " -flto "


	# append the python3 ldflag if it is macOS, don't need it for Linux
	if platform.system() == "Darwin":
		cmd = cmd + "-undefined dynamic_lookup"

	subprocess.check_call(cmd, shell=True)
	spec = importlib.util.spec_from_file_location(module_name, module_filename_so)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	os.unlink(module_filename_cpp)
	os.unlink(module_filename_object)
	module_so_list.append(module_filename_so)

	return module

import atexit
def cleanup_module():
	global module_so_list
	for filename in module_so_list:
		os.unlink(filename)
	module_so_list = []
atexit.register(cleanup_module)
