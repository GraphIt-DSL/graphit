#!/usr/local/bin/python

import unittest
import subprocess
import os
import shutil
import sys

use_parallel = False
use_numa = False

GRAPHIT_BUILD_DIRECTORY = "${GRAPHIT_BUILD_DIRECTORY}".strip().rstrip("/")
GRAPHIT_SOURCE_DIRECTORY = "${GRAPHIT_SOURCE_DIRECTORY}".strip().rstrip("/")
CXX_COMPILER = "${CXX_COMPILER}"


class TestGraphitCompiler(unittest.TestCase):
    first_time_setup = True

    # NOTE: currently the tests can only work within the build/bin directory
    @classmethod
    def setUpClass(cls):
        build_dir = GRAPHIT_BUILD_DIRECTORY
        if not os.path.isdir(build_dir):
            # This can never be true now since the test is run from the build directory
            print ("build the binaries")
            # shutil.rmtree("../../build_dir")
            os.mkdir(build_dir)
            os.chdir(build_dir)
            subprocess.call(["cmake", ".."])
            subprocess.call(["make"])
            os.chdir('./bin')
        else:
            # working directory is in the bin folder
            os.chdir(build_dir)
            os.chdir('./bin')

        cwd = os.getcwd()
        cls.root_test_input_dir = GRAPHIT_SOURCE_DIRECTORY + "/test/input/"
        cls.root_test_input_with_schedules_dir = GRAPHIT_SOURCE_DIRECTORY + "/test/input_with_schedules/"
        cls.compile_flags = "-std=gnu++1y"
        cls.include_path = GRAPHIT_SOURCE_DIRECTORY + "/src/runtime_lib/"
        cls.output_file_name = "test.cpp"
        cls.executable_file_name = "test.o"

        if use_numa:
            cls.numa_flags = " -lnuma -DNUMA -qopenmp"
            cls.cpp_compiler = "icc"
            cls.parallel_framework = "-DOPENMP"
        else:
            cls.numa_flags = ""
            cls.cpp_compiler = CXX_COMPILER
            cls.parallel_framework = "-DCILK"

    def get_command_output(self, command):
        output = ""
        if isinstance(command, list):
            proc = subprocess.Popen(command, stdout=subprocess.PIPE)
        else:
            proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        proc.wait()
        for line in proc.stdout.readlines():
            if isinstance(line, bytes):
                line = line.decode()
            output += line.rstrip() + "\n"
        proc.stdout.close()
        return output

    def setUp(self):
        self.clean_up()

        # def tearDown(self):
        # self.clean_up()

    def clean_up(self):
        # clean up previously generated files
        if os.path.isfile(self.output_file_name):
            os.remove(self.output_file_name)
        if os.path.isfile(self.executable_file_name):
            os.remove(self.executable_file_name)

    # compiles the program with a separate input algorithm file and input schedule file
    # allows us to unit test various different schedules with the same algorithm
    def basic_compile_test_with_separate_algo_schedule_files(self, input_algo_file, input_schedule_file):
        input_algos_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input/'
        input_schedules_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input_with_schedules/'
        print ("current directory: " + os.getcwd())
        algo_file = input_algos_path + input_algo_file
        schedule_file = input_schedules_path + input_schedule_file
        compile_cmd = "python graphitc.py -a " + algo_file + " -f " + schedule_file + " -o test.cpp"
        print (compile_cmd)
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = self.cpp_compiler + " -g -std=gnu++1y -I " + self.include_path + " " + self.numa_flags + " test.cpp -o test.o"
        if use_parallel:
            print ("using icpc for parallel compilation")
            cpp_compile_cmd = "icpc -g -std=gnu++1y -I " + self.include_path + " " + self.parallel_framework + " " + self.numa_flags + " test.cpp -o test.o"
        print (cpp_compile_cmd)
        subprocess.check_call(cpp_compile_cmd, shell=True)

    # compiles the input file with both the algorithm and schedule specification
    def basic_compile_test(self, input_file_name):
        input_with_schedule_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input_with_schedules/'
        print ("current directory: " + os.getcwd())
        compile_cmd = "python graphitc.py -f " + input_with_schedule_path + input_file_name + " -o test.cpp"
        print (compile_cmd)
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = self.cpp_compiler + " -g -std=gnu++1y -I " + self.include_path + " " + self.numa_flags + " test.cpp -o test.o"

        if use_parallel:
            print ("using icpc for parallel compilation")
            cpp_compile_cmd = "icpc -g -std=gnu++1y -I " + self.include_path + " " + self.parallel_framework + " " + self.numa_flags + " test.cpp -o test.o"

        subprocess.check_call(cpp_compile_cmd, shell=True)

    #def basic_compile_exec_test(self, input_file_name):
    #    input_with_schedule_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input_with_schedules/'
    #    print ("current directory: " + os.getcwd())
    #    compile_cmd = "python graphitc.py -f " + input_with_schedule_path + input_file_name + " -o test.cpp"
    #    print (compile_cmd)
    #    subprocess.check_call(compile_cmd, shell=True)
    #    cpp_compile_cmd = self.cpp_compiler + " -g -std=gnu++1y -I " + self.include_path + " " + self.numa_flags + " test.cpp -o test.o"
    #    subprocess.check_call(cpp_compile_cmd, shell=True)
    #    os.chdir("..")
    #    subprocess.check_call("bin/test.o")
    #    os.chdir("bin")

    def basic_compile_exec_test(self, input_file_name, extra_cpp_args=[], extra_exec_args=[]):
        # "-f" and "-o" must not have space in the string, otherwise it doesn't read correctly
        graphit_compile_cmd = ["python", GRAPHIT_BUILD_DIRECTORY + "/bin/graphitc.py", "-f", self.root_test_input_with_schedules_dir + input_file_name, "-o" , self.output_file_name]
        # check the return code of the call as a way to check if compilation happened correctly
        self.assertEqual(subprocess.call(graphit_compile_cmd), 0)
        # check if g++ compilation succeeded
        cpp_compile_cmd = [self.cpp_compiler, self.compile_flags, "-I", self.include_path , self.output_file_name, "-o", self.executable_file_name] + extra_cpp_args
        self.assertEqual(
            subprocess.call(cpp_compile_cmd),
            0)
        self.assertEqual(subprocess.call(["./"+ self.executable_file_name] + extra_exec_args), 0)

    def basic_compile_with_schedules(self, algo_file, schedule_file, extra_cpp_args=[], extra_exec_args=[]):
        # "-f" and "-o" must not have space in the string, otherwise it doesn't read correctly
        graphit_compile_cmd = ["python", GRAPHIT_BUILD_DIRECTORY + "/bin/graphitc.py", "-a", self.root_test_input_dir + algo_file, "-f", self.root_test_input_with_schedules_dir + schedule_file, "-o" , self.output_file_name]
        # check the return code of the call as a way to check if compilation happened correctly
        self.assertEqual(subprocess.call(graphit_compile_cmd), 0)
        # check if g++ compilation succeeded
        cpp_compile_cmd = [self.cpp_compiler, self.compile_flags, "-I", self.include_path , self.output_file_name, "-o", self.executable_file_name] + extra_cpp_args
        self.assertEqual(
            subprocess.call(cpp_compile_cmd),
            0)

    def expect_output_val(self, input_file_name, expected_output_val, extra_cpp_args=[], extra_exec_args=[]):
        self.basic_compile_exec_test(input_file_name, extra_cpp_args, extra_exec_args)
        output = self.get_command_output(["./"+ self.executable_file_name]+extra_exec_args).split("\n")[0]
        print ("output: " + str(output.strip()))
        self.assertEqual(float(output.strip()), expected_output_val)

    def expect_output_val_with_separate_schedule(self, algo_file, schedule_file, expected_output_val, extra_cpp_args=[], extra_exec_args=[]):
        self.basic_compile_with_schedules(algo_file, schedule_file, extra_cpp_args, extra_exec_args)
        output = self.get_command_output(["./"+ self.executable_file_name]+extra_exec_args).split("\n")[0]
        print ("output: " + str(output.strip()))
        self.assertEqual(float(output.strip()), expected_output_val)

    def basic_library_compile(self, input_file_name, input_file_directory='/test/input_with_schedules/',
                              driver='library_test_driver.cpp'):
        input_with_schedule_path = GRAPHIT_SOURCE_DIRECTORY + input_file_directory
        print ("current directory: " + os.getcwd())
        compile_cmd = "python graphitc.py -f " + input_with_schedule_path + input_file_name + " -o test.cpp"
        print (compile_cmd)
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = self.cpp_compiler + " -g -std=gnu++1y -I " + self.include_path + " " + " " + driver + "  -o test.o"
        subprocess.check_call(cpp_compile_cmd, shell=True)

    def basic_library_compile_exec_test(self, input_file_name, input_file_directory='/test/input_with_schedules/'):
        self.basic_library_compile(input_file_name, input_file_directory)
        os.chdir("..")
        subprocess.check_call("bin/test.o")
        os.chdir("bin")

    def library_cf_verified_test(self, input_file_name, input_file_directory='/test/input_with_schedules/'):
        self.basic_library_compile(input_file_name, input_file_directory, driver='library_test_driver_cf.cpp')
        os.chdir("..")
        cmd = "bin/test.o"
        print(cmd)

        # check the value printed to stdout is as expected
        stdout_str = self.get_command_output(cmd).rstrip()
        print ("output : " + stdout_str)
        self.assertEqual(float(stdout_str), 7.49039)
        os.chdir("bin")

    def library_pr_verified_test(self, input_file_name, input_file_directory='/test/input_with_schedules/'):
        self.basic_library_compile(input_file_name, input_file_directory)
        os.chdir("..")
        cmd = "bin/test.o"
        # check the value printed to stdout is as expected
        output = self.get_command_output(cmd)
        print ("output: " + output.strip())
        self.assertEqual(float(output.strip()), 0.00289518)
        os.chdir("bin")

    def library_sssp_verified_test(self, input_file_name, input_file_directory='/test/input_with_schedules/'):
        self.basic_library_compile(input_file_name, input_file_directory, driver='library_test_driver_weighted.cpp')
        os.chdir("..")
        cmd = "bin/test.o > verifier_input"

        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/sssp_verifier -f " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.wel -t verifier_input -r 0"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def astar_verified_test(self, graphit_file, input_file_name, use_separate_algo_file=True, extra_cpp_args=[], extra_exec_args=[]):
        input_algos_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input/'
        input_schedules_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input_with_schedules/'
        print ("current directory: " + os.getcwd())
        if (use_separate_algo_file):
            algo_file = input_algos_path + graphit_file
            graphit_compile_cmd = "python graphitc.py -a " + algo_file + " -f " + input_schedules_path + input_file_name + " -o  test.cpp"
            print (graphit_compile_cmd)
            self.assertEqual(subprocess.call(graphit_compile_cmd, shell=True), 0)
            compile_cpp_cmd = [self.cpp_compiler, self.compile_flags, "-g", "-I", self.include_path,
                               self.output_file_name, "-o", self.executable_file_name] + extra_cpp_args
            print(compile_cpp_cmd)
            # check if g++ compilation succeeded
            self.assertEqual(subprocess.call(compile_cpp_cmd), 0)
            cmd = "./" + self.executable_file_name + " " + extra_exec_args[0] + "> verifier_input"
            print(cmd)
            self.assertEqual(subprocess.call(cmd, shell=True), 0)

            # invoke the PPSP verifier with starting point 0, end point 4
            verify_cmd = "./ppsp_verifier -f " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/monaco.bin -t verifier_input -r 0 -u 4"
            print (verify_cmd)

            output = self.get_command_output(verify_cmd)
            test_flag = False
            for line in output.rstrip().split("\n"):
                if line.rstrip().find("SUCCESSFUL") != -1:
                    test_flag = True
                    break
            self.assertEqual(test_flag, True)
        else:
            print("not supporting default schedules with AStar yet")

    def bfs_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bfs_with_filename_arg.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        os.chdir("..")
        cmd = "OMP_PLACES=sockets ./bin/test.o " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.el" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/bfs_verifier -f " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.el -t verifier_input -r 8"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def cc_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("cc.gt", input_file_name)
        else:
            self.basic_compile_test(
                input_file_name)  # proc = subprocess.Popen(["./"+ self.executable_file_name], stdout=subprocess.PIPE)
        os.chdir("..")
        cmd = "OMP_PLACES=sockets ./bin/test.o" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/cc_verifier -f " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.el -t verifier_input"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def bc_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bc.gt", input_file_name)
        else:
            self.basic_compile_test(
                input_file_name)  # proc = subprocess.Popen(["./"+ self.executable_file_name], stdout=subprocess.PIPE)
        os.chdir("..")
        cmd = "OMP_PLACES=sockets ./bin/test.o " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.el" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/bc_verifier -f ../test/graphs/4.el -t verifier_input -r 3"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def bc_functor_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bc_functor.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)  # proc = subprocess.Popen(["./"+ self.executable_file_name], stdout=subprocess.PIPE)
        os.chdir("..")
        cmd = "OMP_PLACES=sockets ./bin/test.o " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.el" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/bc_verifier -f ../test/graphs/4.el -t verifier_input -r 3"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def sssp_verified_test(self, input_file_name,
                           use_separate_algo_file=True,
                           use_delta_stepping=False,
                           use_delta_from_argv=False):
        if use_separate_algo_file:
            # just use the regular Bellman-Ford based source file
            if not use_delta_stepping:
                self.basic_compile_test_with_separate_algo_schedule_files("sssp.gt", input_file_name)
            # use delta stepping source file
            else:
                self.basic_compile_test_with_separate_algo_schedule_files("delta_stepping.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        os.chdir("..");

        if (use_delta_from_argv):
            cmd = "OMP_PLACES=sockets ./bin/test.o 2" + " > verifier_input"
        else:
            cmd = "OMP_PLACES=sockets ./bin/test.o" + " > verifier_input"

        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the SSSP verifier
        verify_cmd = "./bin/sssp_verifier -f " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.wel -t verifier_input -r 0"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def pr_verified_test(self, input_file_name, use_separate_algo_file=False, use_segment_argv=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("pagerank_with_filename_arg.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)

        if not use_segment_argv:
            cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"
        else:
            cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el  2"
        print (cmd)

        output = self.get_command_output(cmd).split("\n")[0]
        # check the value printed to stdout is as expected
        print ("output: " + output.strip())
        self.assertEqual(float(output.strip()), 0.00289518)

    def bc_basic_compile_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bc.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"

    def bc_functor_basic_compile_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bc_functor.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"

    def pr_delta_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("pr_delta.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"
        print (cmd)
        output = self.get_command_output(cmd)
        # check the value printed to stdout is as expected
        lines = output.strip().split("\n")
        print (lines)
        self.assertEqual(float(lines[0].strip()), 1)
        # first frontier has 5 vertices
        self.assertEqual(float(lines[2].strip()), 5)
        self.assertEqual(float(lines[3].strip()), 0.566667)
        # 2nd frontier has 5 vertices too
        self.assertEqual(float(lines[5].strip()), 5)
        # 3rd frontier has 3 vertices
        self.assertEqual(float(lines[8].strip()), 3)
        # 4th frontier
        self.assertEqual(float(lines[11].strip()), 2)
        # sum of delta at 4th iter
        self.assertEqual(float(lines[12].strip()), 0.0261003)
        # 5th frontier
        self.assertEqual(float(lines[14].strip()), 1)

    def cf_verified_test(self, input_file_name, use_separate_algo_file=False):
        if (use_separate_algo_file):
            self.basic_compile_test_with_separate_algo_schedule_files("cf.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test_cf.wel"
        print (cmd)
        # check the value printed to stdout is as expected
        output = self.get_command_output(cmd).strip().split("\n")[0]
        print ("output: " + output.strip())
        self.assertEqual(float(output.strip()), 7.49039)

    def eigenvector_centrality_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("eigenvector_centrality.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"
        print (cmd)
        # check the value printed to stdout is as expected
        lines = self.get_command_output(cmd).strip().split("\n")
        print (lines)
        self.assertEqual(float(lines[0].strip()), 3.2)

    def closeness_centrality_unweighted_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("closeness_centrality_unweighted.gt",
                                                                      input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"
        print (cmd)
        # check the value printed to stdout is as expected
        lines = self.get_command_output(cmd).strip().split("\n")
        print (lines)
        self.assertEqual(float(lines[3].strip()), 3)

    def closeness_centrality_weighted_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("closeness_centrality_weighted.gt",
                                                                      input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"
        print (cmd)
        # check the value printed to stdout is as expected
        lines = self.get_command_output(cmd).strip().split("\n")
        print (lines)
        self.assertEqual(float(lines[1].strip()), 15)

    def closeness_centrality_unweighted_functor_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("closeness_centrality_unweighted_functor.gt",
                                                                          input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"
        print (cmd)
        # check the value printed to stdout is as expected
        lines = self.get_command_output(cmd).strip().split("\n")
        print (lines)
        self.assertEqual(float(lines[1].strip()), 3)

    def closeness_centrality_weighted_functor_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("closeness_centrality_weighted_functor.gt",
                                                                      input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/test.el"
        print (cmd)
        # check the value printed to stdout is as expected
        lines = self.get_command_output(cmd).strip().split("\n")
        print (lines)
        self.assertEqual(float(lines[1].strip()), 15)

    def ppsp_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("ppsp_delta_stepping.gt",
                                                                      input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./" + self.executable_file_name + " " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.wel  > verifier_input"
        print (cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        proc.wait()

        # invoke the PPSP verifier with starting point 0, end point 4
        verify_cmd = "./ppsp_verifier -f " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4.wel -t verifier_input -r 0 -u 4"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break
        self.assertEqual(test_flag, True)

    def tc_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("tc.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        os.chdir("..")
        cmd = "OMP_PLACES=sockets ./bin/test.o " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4_sym.el" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        verify_cmd = "./bin/tc_verifier -s -f " + GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/4_sym.el -t verifier_input"
        print (verify_cmd)
        output = self.get_command_output(verify_cmd)
        test_flag = False
        for line in output.rstrip().split("\n"):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")


    def test_simple_splitting(self):
        self.basic_compile_test("simple_loop_index_split.gt")

    def test_pagerank_AoS(self):
        # self.basic_compile_test("simple_pagerank_with_AoS.gt")
        self.basic_compile_test_with_separate_algo_schedule_files("pagerank.gt", "simple_pagerank_with_AoS.gt")

    def test_filter_sum_parallel(self):
        self.basic_compile_test_with_separate_algo_schedule_files("simple_apply_sum.gt", "simple_vector_sum.gt")

    def test_sssp_compile_runtime_delta_parameter(self):
        self.basic_compile_test_with_separate_algo_schedule_files("delta_stepping.gt", "SparsePush_VertexParallel_Delta_argv.gt")

    def test_eigenvector_pagerank_fusion(self):
        self.basic_compile_test("eigenvector_pr_fusion.gt")

    def test_eigenvector_pagerank_segment(self):
        self.basic_compile_test("eigenvector_pr_segment.gt")

    def test_bfs_push_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_push_parallel_cas.gt", True)

    def test_bfs_hybrid_dense_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_hybrid_dense_parallel_cas.gt", True)

    def test_bfs_hybrid_dense_parallel_cas_segment_verified(self):
        self.bfs_verified_test("bfs_hybrid_dense_parallel_cas_segment.gt", True)

    def test_bfs_push_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_push_parallel_cas.gt", True)

    def test_bfs_pull_parallel_verified(self):
        self.bfs_verified_test("bfs_pull_parallel.gt", True)

    def test_bfs_pull_edge_aware_parallel_verified(self):
        self.bfs_verified_test("bfs_pull_edge_aware_parallel.gt", True)

    def test_bfs_pull_parallel_segment_verified(self):
        self.bfs_verified_test("bfs_pull_parallel_segment.gt", True)

    def test_bfs_push_sliding_queue_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_push_sliding_queue_parallel_cas.gt", True)

    def test_cc_hybrid_dense_parallel_cas_verified(self):
        self.cc_verified_test("cc_hybrid_dense_parallel_cas.gt", True)

    def test_cc_hybrid_dense_parallel_bitvector_verified(self):
        self.cc_verified_test("cc_hybrid_dense_parallel_bitvector.gt", True)

    def test_cc_hybrid_dense_parallel_bitvector_segment_verified(self):
        self.cc_verified_test("cc_hybrid_dense_parallel_bitvector_segment.gt", True)

    def test_cc_hybrid_dense_parallel_bitvector_numa_verified(self):
        if self.numa_flags:
            self.cc_verified_test("cc_hybrid_dense_parallel_bitvector_numa.gt", True)

    def test_cc_push_parallel_cas_verified(self):
        self.cc_verified_test("cc_push_parallel_cas.gt", True)

    def test_cc_pull_parallel_verified(self):
        self.cc_verified_test("cc_pull_parallel.gt", True)

    def test_cc_pull_parallel_segment_verified(self):
        self.cc_verified_test("cc_pull_parallel_segment.gt", True)

    def test_cc_pull_parallel_numa_verified(self):
        if self.numa_flags:
            self.cc_verified_test("cc_pull_parallel_numa.gt", True)

    def test_sssp_push_parallel_cas_verified(self):
        self.sssp_verified_test("sssp_push_parallel_cas.gt", True)

    def test_sssp_hybrid_denseforward_parallel_cas_verified(self):
        self.sssp_verified_test("sssp_hybrid_denseforward_parallel_cas.gt", True)

    def test_sssp_hybrid_dense_parallel_cas_verified(self):
        self.sssp_verified_test("sssp_hybrid_dense_parallel_cas.gt", True)

    def test_sssp_pull_parallel_verified(self):
        self.sssp_verified_test("sssp_pull_parallel.gt", True)

    def test_sssp_push_parallel_sliding_queue_verified(self):
        self.sssp_verified_test("sssp_push_parallel_sliding_queue.gt", True)

    def test_pagerank_parallel_pull_expect(self):
        self.pr_verified_test("pagerank_pull_parallel.gt", True)

    def test_pagerank_parallel_hybrid_dense_expect(self):
        self.pr_verified_test("pagerank_hybrid_dense.gt", True)

    def test_pagerank_parallel_push_expect(self):
        self.pr_verified_test("pagerank_push_parallel.gt", True)

    def test_pagerank_parallel_pull_load_balance_expect(self):
        self.pr_verified_test("pagerank_pull_parallel_load_balance.gt", True)

    def test_pagerank_parallel_pull_segment_expect(self):
        self.pr_verified_test("pagerank_pull_parallel_segment.gt", True)

    def test_pagerank_parallel_pull_segment_argv_expect(self):
        self.pr_verified_test("pagerank_pull_parallel_segment_argv.gt", True, True)

    def test_pagerank_parallel_pull_numa_expect(self):
        if self.numa_flags:
            self.pr_verified_test("pagerank_pull_parallel_numa.gt", True)

    def test_pagerank_parallel_pull_numa_expect(self):
        if self.numa_flags:
            self.pr_verified_test("pagerank_pull_parallel_numa_one_seg.gt", True)

    def test_cf_parallel_expect(self):
        self.cf_verified_test("cf_pull_parallel.gt", True)

    def test_cf_parallel_segment_expect(self):
        self.cf_verified_test("cf_pull_parallel_segment.gt", True)

    def test_cf_parallel_load_balance_expect(self):
        self.cf_verified_test("cf_pull_parallel_load_balance.gt", True)

    def test_cf_parallel_load_balance_segment_expect(self):
        self.cf_verified_test("cf_pull_parallel_load_balance_segment.gt", True)

    def test_prdelta_parallel_pull(self):
        self.pr_delta_verified_test("pagerank_delta_pull_parallel.gt", True)

    def test_prdelta_parallel_sparse_push(self):
        self.pr_delta_verified_test("pagerank_delta_sparse_push_parallel.gt", True)

    def test_prdelta_parallel_pull_segment_expect(self):
        self.pr_delta_verified_test("pagerank_delta_pull_parallel_segment.gt", True)

    def test_prdelta_parallel_pull_numa_expect(self):
        if self.numa_flags:
            self.pr_delta_verified_test("pagerank_delta_pull_parallel_numa.gt", True)

    def test_prdelta_hybrid_alternate_direction_specification(self):
        self.pr_delta_verified_test("pagerank_delta_hybrid_dense_parallel_bitvector_altdir.gt", True)

    def test_prdelta_parallel_hybrid_segment_expect(self):
        self.pr_delta_verified_test("pagerank_delta_hybrid_dense_parallel_segment.gt", True)

    def test_prdelta_parallel_hybrid_numa_expect(self):
        if self.numa_flags:
            self.pr_delta_verified_test("pagerank_delta_hybrid_dense_parallel_numa.gt", True)

    def test_prdelta_parallel_load_balance_pull(self):
        self.pr_delta_verified_test("pagerank_delta_pull_parallel_load_balance.gt", True)

    def test_prdelta_parallel_load_balance_hybrid_dense_with_bitvec(self):
        self.pr_delta_verified_test("pagerank_delta_hybrid_dense_parallel_bitvector.gt", True)

    def test_prdelta_parallel_load_balance_hybrid_dense_without_bitvec(self):
        self.pr_delta_verified_test("pagerank_delta_hybrid_dense_parallel_load_balance_no_bitvector.gt", True)

    def test_eigenvector_centrality_densepull_parallel(self):
        self.eigenvector_centrality_verified_test("eigenvector_centrality_DensePull_parallel.gt", True)

    def test_closeness_centrality_unweighted_hybrid_parallel(self):
        self.closeness_centrality_unweighted_test("closeness_centrality_unweighted_hybrid_parallel.gt", True)

    def test_closeness_centrality_weighted_hybrid_parallel(self):
        self.closeness_centrality_weighted_test("closeness_centrality_weighted_hybrid_parallel.gt", True)

    def test_closeness_centrality_unweighted_functor_hybrid_parallel(self):
        self.closeness_centrality_unweighted_functor_test("closeness_centrality_unweighted_hybrid_parallel.gt", True)

    def test_closeness_centrality_weighted_functor_hybrid_parallel(self):
        self.closeness_centrality_weighted_functor_test("closeness_centrality_weighted_hybrid_parallel.gt", True)

    def test_bc_SparsePushDensePull_basic(self):
        self.bc_basic_compile_test("bc_SparsePushDensePull.gt", True);

    def test_bc_SparsePushDensePull_bitvector_basic(self):
        self.bc_basic_compile_test("bc_SparsePushDensePull_bitvector.gt", True);

    def test_bc_SparsePush_basic(self):
        self.bc_basic_compile_test("bc_SparsePush.gt", True);

    def test_bc_SparsePushDensePull_bitvector_cache_basic(self):
        self.bc_basic_compile_test("bc_SparsePushDensePull_bitvector_cache.gt", True);

    def test_bc_SparsePush_verified(self):
        self.bc_verified_test("bc_SparsePush.gt", True)

    def test_bc_SparsePushDensePull_verified(self):
        self.bc_verified_test("bc_SparsePushDensePull.gt", True)

    def test_bc_SparsePushDensePull_bitvector_verified(self):
        self.bc_verified_test("bc_SparsePushDensePull_bitvector.gt", True)

    def test_bc_SparsePushDensePull_bitvector_cache_verified(self):
        self.bc_verified_test("bc_SparsePushDensePull_bitvector_cache.gt", True)

    def test_bc_functor_SparsePushDensePull_basic(self):
        self.bc_functor_basic_compile_test("bc_SparsePushDensePull.gt", True);

    def test_bc_functor_SparsePushDensePull_bitvector_basic(self):
        self.bc_functor_basic_compile_test("bc_SparsePushDensePull_bitvector.gt", True);

    def test_bc_functor_SparsePush_basic(self):
        self.bc_functor_basic_compile_test("bc_SparsePush.gt", True);

    def test_bc_functor_SparsePushDensePull_bitvector_cache_basic(self):
        self.bc_functor_basic_compile_test("bc_SparsePushDensePull_bitvector_cache.gt", True);

    def test_bc_functor_SparsePush_verified(self):
        self.bc_functor_verified_test("bc_SparsePush.gt", True)

    def test_bc_functor_SparsePushDensePull_verified(self):
        self.bc_functor_verified_test("bc_SparsePushDensePull.gt", True)

    def test_bc_functor_SparsePushDensePull_bitvector_verified(self):
        self.bc_functor_verified_test("bc_SparsePushDensePull_bitvector.gt", True)

    def test_bc_functor_SparsePushDensePull_bitvector_cache_verified(self):
        self.bc_functor_verified_test("bc_SparsePushDensePull_bitvector_cache.gt", True)

    def test_delta_stepping_SparsePush_schedule(self):
        self.sssp_verified_test("SparsePush_VertexParallel.gt", True, True)

    def test_delta_stepping_DensePull_schedule(self):
        self.sssp_verified_test("DensePull_VertexParallel.gt", True, True)

    def test_delta_stepping_SparsePushDensePull_schedule(self):
        self.sssp_verified_test("SparsePushDensePull_VertexParallel.gt", True, True)

    def test_delta_stepping_SparsePush_delta2_schedule(self):
        self.sssp_verified_test("SparsePush_VertexParallel_Delta2.gt", True, True)

    def test_delta_stepping_eager_no_merge(self):
        self.sssp_verified_test("priority_update_eager_no_merge.gt", True, True);

    def test_delta_stepping_eager_no_merge_argv1(self):
        self.sssp_verified_test("priority_update_eager_no_merge_argv1.gt", True, True, True);

    def test_delta_stepping_eager_with_merge(self):
        self.sssp_verified_test("priority_update_eager_with_merge.gt", True, True);

    def test_ppsp_delta_stepping_eager_no_merge(self):
        self.ppsp_verified_test("priority_update_eager_no_merge.gt", True);

    def test_ppsp_delta_stepping_SparsePush_parallel(self):
        self.ppsp_verified_test("SparsePushDensePull_VertexParallel.gt", True);

    def test_ppsp_delta_stepping_SparsePush_parallel_delta2(self):
        self.ppsp_verified_test("SparsePush_VertexParallel_Delta2.gt", True);

    def test_tc_hiroshi(self):
        self.tc_verified_test("tc_hiroshi.gt", True);

    def test_tc_multi_skip(self):
        self.tc_verified_test("tc_multiskip.gt", True);

    def test_tc_naive(self):
        self.tc_verified_test("tc_naive.gt", True);

    def test_tc_empty(self):
        self.tc_verified_test("tc_empty.gt", True);

    def test_delta_stepping_eager_with_merge(self):
        self.ppsp_verified_test("priority_update_eager_with_merge.gt", True);

    def test_astar_eager_with_merge(self):
        self.astar_verified_test("astar.gt",
                                 "priority_update_eager_with_merge.gt",
                                 True,
                                 [self.root_test_input_dir + "astar_distance_loader.cpp"],
                                 [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/monaco.bin"]);

    def test_astar_sparsepush_parallel(self):
        self.astar_verified_test("astar.gt",
                                 "SparsePush_VertexParallel.gt",
                                 True,
                                 [self.root_test_input_dir + "astar_distance_loader.cpp"],
                                 [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/monaco.bin"]);

    def test_astar_sparsepush_parallel_delta2(self):
        self.astar_verified_test("astar.gt",
                                 "SparsePush_VertexParallel_Delta2.gt",
                                 True,
                                 [self.root_test_input_dir + "astar_distance_loader.cpp"],
                                 [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/monaco.bin"]);

    def test_astar_eager_with_merge_functor(self):
        self.astar_verified_test("astar_functor.gt",
                                 "priority_update_eager_with_merge.gt",
                                 True,
                                 [self.root_test_input_dir + "astar_distance_loader.cpp"],
                                 [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/monaco.bin"]);

    def test_astar_sparsepush_parallel_functor(self):
        self.astar_verified_test("astar_functor.gt",
                                 "SparsePush_VertexParallel.gt",
                                 True,
                                 [self.root_test_input_dir + "astar_distance_loader.cpp"],
                                 [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/monaco.bin"]);

    def test_astar_sparsepush_parallel_delta2_functor(self):
        self.astar_verified_test("astar_functor.gt",
                                 "SparsePush_VertexParallel_Delta2.gt",
                                 True,
                                 [self.root_test_input_dir + "astar_distance_loader.cpp"],
                                 [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/monaco.bin"]);

    def test_k_core_unordered_sparsepush(self):
        self.expect_output_val_with_separate_schedule("unordered_kcore.gt", "SparsePush_VertexParallel.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100.el"])

    def test_k_core_unordered_sparsepush_densepull(self):
        self.expect_output_val_with_separate_schedule("unordered_kcore.gt", "SparsePushDensePull_VertexParallel.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100.el"])


    def test_k_core_uint_const_sum_reduce(self):
        self.expect_output_val_with_separate_schedule("k_core_uint.gt", "k_core_const_sum_reduce.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100"])

    def test_k_core_const_sum_reduce(self):
        self.expect_output_val_with_separate_schedule("k_core.gt", "k_core_const_sum_reduce.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100"])

    def test_k_core_sparsepush(self):
        self.expect_output_val_with_separate_schedule("k_core.gt", "SparsePush_VertexParallel.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100.el"])

    def test_k_core_uint_sparsepush(self):
        self.expect_output_val_with_separate_schedule("k_core_uint.gt", "SparsePush_VertexParallel.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100.el"])

    def test_k_core_uint_sparsepush_16_open_buckets(self):
        self.expect_output_val_with_separate_schedule("k_core_uint.gt", "KCore_SparsePush_VertexParallel_16_Open.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100.el"])

    def test_k_core_uint_sparsepush_16_open_buckets_rmat10(self):
        self.expect_output_val_with_separate_schedule("k_core_uint.gt", "KCore_SparsePush_VertexParallel_16_Open.gt", 38, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rmat10.el"])


    def test_k_core_sparsepush_densepull(self):
        self.expect_output_val_with_separate_schedule("k_core.gt", "SparsePushDensePull_VertexParallel.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100.el"])

    def test_k_core_densepull_parallel(self):
        self.expect_output_val_with_separate_schedule("k_core.gt", "DensePull_VertexParallel.gt", 4, [], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100.el"])

    def test_set_cover(self):
        self.expect_output_val("set_cover.gt", 33, [GRAPHIT_SOURCE_DIRECTORY+"/test/input_with_schedules/set_cover_extern.cpp"], [GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/rMatGraph_J_5_100"]) 

    def test_basic_library(self):
        self.basic_library_compile_exec_test("export_simple_edgeset_apply.gt");

    def test_library_pagerank(self):
        self.basic_library_compile_exec_test("export_pr.gt", '/test/input/');

    def test_library_pagerank_verified(self):
        self.library_pr_verified_test("export_pr.gt", '/test/input/');

    def test_library_pagerank_with_return_verified(self):
        self.library_pr_verified_test("export_pr_with_return.gt", '/test/input/');

    def test_library_sssp_with_return_verified(self):
        self.library_sssp_verified_test("export_sssp.gt", '/test/input/');

    def test_library_cf_with_return_basic_compile(self):
        self.basic_library_compile("export_cf_vector_input_with_return.gt", '/test/input/',
                                   driver='library_test_driver_weighted.cpp')

    def test_library_cf_with_return_verified(self):
        self.library_cf_verified_test("export_cf_vector_input_with_return.gt", '/test/input/')


if __name__ == '__main__':

    #while len(sys.argv) > 1:
    #    if "parallel" in sys.argv:
    #        use_parallel = True
    #        print ("using parallel")
    #        del sys.argv[sys.argv.index("parallel")]
    #    if "numa" in sys.argv:
    #        use_numa = True
    #        print ("using numa")
    #        del sys.argv[sys.argv.index("numa")]
    for arg in sys.argv:
        if "parallel" == arg:
            use_parallel = True
            print ("using parallel")
        elif "numa" == arg:
            use_numa = True
            print ("using numa")
        

    # comment out if want to enable a specific test only
    unittest.main(verbosity=2)

    # used for enabling a specific test

    #suite = unittest.TestSuite()
    #suite.addTest(TestGraphitCompiler('test_astar_eager_with_merge_functor'))
    #unittest.TextTestRunner(verbosity=2).run(suite)