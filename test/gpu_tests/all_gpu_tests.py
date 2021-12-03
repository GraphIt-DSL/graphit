import unittest
import subprocess
import os
import shutil
import sys

GRAPHIT_BUILD_DIRECTORY="${GRAPHIT_BUILD_DIRECTORY}".strip().rstrip("/")
GRAPHIT_SOURCE_DIRECTORY="${GRAPHIT_SOURCE_DIRECTORY}".strip().rstrip("/")
CXX_COMPILER="${CXX_COMPILER}"

NVCC_COMPILER="${NVCC_COMPILER}"

class TestGPURuntimeLibrary(unittest.TestCase):
	@classmethod
	def get_command_output_class(self, command):
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

	def get_command_output(self, command):
		(exitcode, output) = self.get_command_output_class(command)
		self.assertEqual(exitcode, 0)
		return output

	def sssp_verified_test(self, input_file_name, use_delta=False):
		self.cpp_compile_test(input_file_name, [])
		if use_delta:
			#start point 0, delta 10, verified
			self.get_command_output(self.executable_name + " " + self.graph_directory + "/4.wel 0 10 v > " + self.verifier_input)
		else:
			self.get_command_output(self.executable_name + " " + self.graph_directory + "/4.wel 0 v > " + self.verifier_input)	     
		output = self.get_command_output(self.verifier_directory + "/sssp_verifier -f " + self.graph_directory +  "/4.wel -t " + self.verifier_input + "  -r 0")		
		test_flag = False
		for line in output.rstrip().split("\n"):
			if line.rstrip().find("SUCCESSFUL") != -1:
				test_flag = True
				break;
		self.assertEqual(test_flag, True)
		
	@classmethod	
	def setUpClass(cls):
		if NVCC_COMPILER == "CUDA_NVCC_EXECUTABLE-NOTFOUND":
			print ("Cannot find CUDA compiler")
			exit(-1)	

		cls.build_directory = GRAPHIT_BUILD_DIRECTORY
		cls.scratch_directory = GRAPHIT_BUILD_DIRECTORY + "/scratch"
		cls.verifier_directory = cls.build_directory + "/bin"	
		if os.path.isdir(cls.scratch_directory):
			shutil.rmtree(cls.scratch_directory)
		os.mkdir(cls.scratch_directory)
		
		cls.nvcc_command = NVCC_COMPILER + " -ccbin " + CXX_COMPILER + " "
		cls.test_input_directory = GRAPHIT_SOURCE_DIRECTORY + "/test/gpu_tests/test_input"
		
		cls.get_command_output_class(cls.nvcc_command + cls.test_input_directory + "/obtain_gpu_cc.cu -o " + cls.scratch_directory + "/obtain_gpu_cc")
		output = cls.get_command_output_class(cls.scratch_directory + "/obtain_gpu_cc")[1].split()

		if len(output) != 2:
			print ("Cannot obtain GPU information")
			exit(-1)
		compute_capability = output[0]
		num_of_sm = output[1]
		
		cls.nvcc_command += " -rdc=true -DNUM_CTA=" + str(int(num_of_sm)*2) + " -DCTA_SIZE=512 -gencode arch=compute_" + compute_capability + ",code=sm_" + compute_capability
		cls.nvcc_command += " -std=c++11 -O3 -I " + GRAPHIT_SOURCE_DIRECTORY + "/src/runtime_lib/ -Xcompiler \"-w\" -Wno-deprecated-gpu-targets --use_fast_math -Xptxas \" -dlcm=ca --maxrregcount=64\" "
		
		shutil.copytree(GRAPHIT_SOURCE_DIRECTORY + "/test/graphs", cls.scratch_directory + "/graphs")
		cls.graph_directory = cls.scratch_directory + "/graphs"
		cls.executable_name = cls.scratch_directory + "/test_executable"	
		cls.cuda_filename = cls.scratch_directory + "/test_cpp.cu"
		
		cls.graphitc_py = GRAPHIT_BUILD_DIRECTORY + "/bin/graphitc.py"
		cls.verifier_input = cls.scratch_directory + "/verifier_input"

	def cpp_compile_test(self, input_file_name, extra_cpp_args=[]):
		if input_file_name[0] == "/":
			compile_command = self.nvcc_command + input_file_name + " -o " + self.executable_name + " " + " ".join(extra_cpp_args)
		else:
			compile_command = self.nvcc_command + self.test_input_directory + "/" + input_file_name + " -o " + self.executable_name + " " + " ".join(extra_cpp_args)
		self.get_command_output(compile_command)
	
	def cpp_exec_test(self, input_file_name, extra_cpp_args=[], extra_exec_args=[]):
		self.cpp_compile_test(input_file_name, extra_cpp_args)
		return self.get_command_output(self.executable_name + " " + " ".join(extra_exec_args))

	def graphit_generate_test(self, input_file_name, input_schedule_name=""):
		if input_file_name[0] != "/":
			input_file_name = self.test_input_directory + "/" + input_file_name
		if input_schedule_name != "" and input_schedule_name[0] != "/":
			input_schedule_name = self.test_input_directory + "/" + input_schedule_name

		if input_schedule_name != "":
			self.get_command_output("python " + self.graphitc_py + " -a " + input_file_name + " -f " + input_schedule_name + " -o " + self.cuda_filename)
		else:
			self.get_command_output("python " + self.graphitc_py + " -f " + input_file_name + " -o " + self.cuda_filename)
		
	def graphit_compile_test(self, input_file_name, input_schedule_name="", extra_cpp_args=[]):	
		self.graphit_generate_test(input_file_name, input_schedule_name)
		self.cpp_compile_test(self.cuda_filename, extra_cpp_args)

	def graphit_exec_test(self, input_file_name, input_schedule_name="", extra_cpp_args=[], extra_exec_args=[]):
		self.graphit_generate_test(input_file_name, input_schedule_name)
		return self.cpp_exec_test(self.cuda_filename, extra_cpp_args, extra_exec_args)
			
	def test_basic_compile(self):
		self.cpp_compile_test("basic_compile.cu")
	def test_basic_load_graph(self):
		output = self.cpp_exec_test("basic_load_graph.cu", [], [self.graph_directory + "/simple_mtx.mtx"])
		output = output.split("\n")
		self.assertEqual(len(output), 2)
		self.assertEqual(output[0], "14, 106")
	def test_runtime_library(self):
		print (self.cpp_exec_test("runtime_lib_tests.cu", ["-I", GRAPHIT_SOURCE_DIRECTORY+"/test/gtest", GRAPHIT_SOURCE_DIRECTORY+"/test/gtest/gtest-all.cc"], [self.graph_directory]))
		
	def test_sssp_lp_runtime_lib(self):
		self.cpp_exec_test("sssp_lp.cu", [], [self.graph_directory + "/simple_mtx.mtx", "v"])

	def test_sssp_lp_verified(self):
		self.sssp_verified_test("sssp_lp.cu")
		
	def test_sssp_delta_stepping(self):
		self.cpp_exec_test("sssp_delta_stepping.cu", [], [self.graph_directory + "/simple_mtx.mtx", "0", "10",  "v"])

	def test_sssp_delta_stepping_verified(self):
		self.sssp_verified_test("sssp_delta_stepping.cu", True)

	def test_sssp_delta_stepping_verified_frontier_byval(self):
		self.sssp_verified_test("sssp_delta_stepping_frontier_byval.cu", True)

	def test_simple_graphit_exec(self):
		output = self.graphit_exec_test("simple_graph_load.gt", "default_gpu_schedule.gt", [], [self.graph_directory + "/simple_mtx.mtx"])
		output = output.split("\n")
		self.assertEqual(len(output), 2)
		self.assertEqual(output[0], "14")

	def test_simple_graphit_sssp_basic_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_default_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_TWCE_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_TWCE_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_TWC_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_TWC_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_CM_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_CM_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_WM_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_WM_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)
		
	def test_simple_graphit_sssp_strict_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_strict_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_vertex_based_kernel_fusion_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_vertex_based_kernel_fusion_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_TWC_kernel_fusion_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_TWC_kernel_fusion_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_TWCE_kernel_fusion_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_TWCE_kernel_fusion_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_CM_kernel_fusion_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_CM_kernel_fusion_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_WM_kernel_fusion_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_WM_kernel_fusion_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

	def test_simple_graphit_sssp_strict_kernel_fusion_schedule(self):
		self.graphit_generate_test("inputs/sssp.gt", "schedules/sssp_strict_kernel_fusion_schedule.gt")
		self.sssp_verified_test(self.cuda_filename, False)

if __name__ == '__main__':
	unittest.main()
	#suite = unittest.TestSuite()
	#suite.addTest(TestGraphitCompiler('test_sssp_delta_stepping'))
	#unittest.TextTestRunner(verbosity=2).run(suite)
