import os
import tempfile
import subprocess
import importlib.util
import platform
import pybind11
import sys

GRAPHIT_BUILD_DIRECTORY="${GRAPHIT_BUILD_DIRECTORY}".strip().rstrip("/")
GRAPHIT_SOURCE_DIRECTORY="${GRAPHIT_SOURCE_DIRECTORY}".strip().rstrip("/")
CXX_COMPILER="${CXX_COMPILER}".strip().rstrip("/")

PARALLEL_NONE=0
PARALLEL_CILK=1
PARALLEL_OPENMP=2

module_so_list = []

def compile_and_load(graphit_source_file, extern_cpp_files=[], linker_args=[], parallelization_type=PARALLEL_NONE):
	graphit_source_file = os.path.expanduser(graphit_source_file.strip())
	# Obtain a unique filename for the module
	#module_file = tempfile.NamedTemporaryFile()
	#module_filename_base = module_file.name
	#module_file.close()
	module_filename_base = os.path.splitext(graphit_source_file)[0]
	module_name = os.path.basename(module_filename_base)    
	module_filename_base = "/tmp/" + module_name

	# compile the file into a cpp file
	module_filename_cpp = module_filename_base + ".cpp"
	module_filename_object = module_filename_base + ".o"
	module_filename_so = module_filename_base + ".so"

	try:
		subprocess.check_call("python " + GRAPHIT_BUILD_DIRECTORY + "/bin/graphitc.py -f " + graphit_source_file + " -o " + module_filename_cpp + " -m " + module_name, stderr=subprocess.STDOUT, shell=True)
	except subprocess.CalledProcessError as e:
		print(e.output)
		return


	compile_command = CXX_COMPILER + " -I" + pybind11.get_include() + " $(python3-config --includes) -c -I " +  GRAPHIT_SOURCE_DIRECTORY + "/src/runtime_lib/ -std=c++14 -DGEN_PYBIND_WRAPPERS -flto -fno-fat-lto-objects -fPIC -fvisibility=hidden -O3 "
	# now compile the file into .so

	if parallelization_type == PARALLEL_CILK:
		compile_command += " -DCILK -fcilkplus "
	elif parallelization_type == PARALLEL_OPENMP:
		compile_command += " -DOPENMP -fopenmp "
	
	try:
		subprocess.check_call(compile_command + module_filename_cpp + " -o " + module_filename_object, shell=True)
	except subprocess.CalledProcessError as e:
		print(e.output)
		return

	extern_objects = []
	for extern_file in extern_cpp_files:
		object_filename = extern_file + ".o"
		subprocess.check_call(compile_command + extern_file + " -o " + object_filename, shell=True)
		extern_objects.append(object_filename)
	
	object_list = " " + " ".join(extern_objects) + " "
	cmd = CXX_COMPILER + " -fPIC -shared -o " + module_filename_so + " " + module_filename_object + " -flto " + object_list


	# append the python3 ldflag if it is macOS, don't need it for Linux
	if platform.system() == "Darwin":
		cmd = cmd + "-undefined dynamic_lookup"
	
	if parallelization_type == PARALLEL_CILK:
		cmd += " -fcilkplus "
	elif parallelization_type == PARALLEL_OPENMP:
		cmd += " -fopenmp "
			
	if len(linker_args) > 0:
		cmd += " " + " ".join(linker_args) + " "
	subprocess.check_call(cmd, shell=True)
	spec = importlib.util.spec_from_file_location(module_name, module_filename_so)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	os.unlink(module_filename_object)
	for extern_object in extern_objects:
		os.unlink(extern_object)
	module_so_list.append(module_filename_so)

	return module

def compile_and_load_cache(graphit_source_file, extern_cpp_files=[], linker_args=[], parallelization_type=PARALLEL_NONE):
	graphit_source_file = os.path.expanduser(graphit_source_file.strip())
	# Obtain a unique filename for the module
	#module_file = tempfile.NamedTemporaryFile()
	#module_filename_base = module_file.name
	#module_file.close()
	module_filename_base = os.path.splitext(graphit_source_file)[0]
	module_name = os.path.basename(module_filename_base)    
	module_filename_base = "/tmp/" + module_name

	# compile the file into a cpp file
	module_filename_cpp = module_filename_base + ".cpp"
	module_filename_object = module_filename_base + ".o"
	module_filename_so = module_filename_base + ".so"

	# check if it is loaded before and it is the latest, we return the cached version.
	if os.path.exists(module_filename_so) and os.stat(graphit_source_file).st_mtime < os.stat(module_filename_so).st_mtime:
		spec = importlib.util.spec_from_file_location(module_name, module_filename_so)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		return module
	
	#else we just follow the normal routine
	return compile_and_load(graphit_source_file, extern_cpp_files, linker_args, parallelization_type)
	

from scipy.sparse import csr_matrix
def read_adjacency_tsv(file):
	"""Read a graph from a file-like object in "adjacency TSV" format,
	returning a `csr_matrix` object.
	In this format (popularized by the MIT GraphChallenge datasets),
	each row has three numbers, separated by tabs: the vertex indices
	and the edge weight.
	"""
	srcs = []
	dests = []
	values = []
	for line in file:
		line = line.strip()
		i, j, v = [int(n) for n in line.split('\t')]
		srcs.append(i)
		dests.append(j)
		values.append(v)
	return csr_matrix((values, (srcs, dests)))

import atexit
def cleanup_module():
	global module_so_list
	for filename in module_so_list:
		if os.path.exists(filename):
			os.unlink(filename)
	module_so_list = []
atexit.register(cleanup_module)
