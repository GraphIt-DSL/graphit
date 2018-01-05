#!/usr/local/bin/python

import unittest
import subprocess
import os
import shutil
import sys

use_parallel = False

class TestGraphitCompiler(unittest.TestCase):
    first_time_setup = True

    #NOTE: currently the tests can only work within the build/bin directory
    @classmethod
    def setUpClass(cls):
        build_dir = "../../build"
        if not os.path.isdir(build_dir):
            print "build the binaries"
            #shutil.rmtree("../../build_dir")
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

        cls.root_test_input_dir = "../test/input/"
        cls.cpp_compiler = "g++"
        cls.compile_flags = "-std=c++11"
        cls.include_path = "../src/runtime_lib"
        cls.output_file_name = "test.cpp"
        cls.executable_file_name = "test.o"


    def setUp(self):
        self.clean_up()

        #def tearDown(self):
        #self.clean_up()

    def clean_up(self):
        #clean up previously generated files
        if os.path.isfile(self.output_file_name):
            os.remove(self.output_file_name)
        if os.path.isfile(self.executable_file_name):
            os.remove(self.executable_file_name)


    # compiles the program with a separate input algorithm file and input schedule file
    # allows us to unit test various different schedules with the same algorithm
    def basic_compile_test_with_separate_algo_schedule_files(self, input_algo_file, input_schedule_file):
        input_algos_path = '../../test/input/'
        input_schedules_path = '../../test/input_with_schedules/'
        print os.getcwd()
        algo_file = input_algos_path + input_algo_file
        schedule_file = input_schedules_path + input_schedule_file
        compile_cmd = "python graphitc.py -a " + algo_file + " -f " + schedule_file + " -o test.cpp"
        print compile_cmd
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = "g++ -g -std=c++11 -I ../../src/runtime_lib/  test.cpp -o test.o"
        if use_parallel:
            print "using icpc for parallel compilation"
            cpp_compile_cmd = "icpc -g -std=c++11 -I ../../src/runtime_lib/ -DCILK test.cpp -o test.o"
        subprocess.check_call(cpp_compile_cmd, shell=True)

    # compiles the input file with both the algorithm and schedule specification
    def basic_compile_test(self, input_file_name):
        input_with_schedule_path = '../../test/input_with_schedules/'
        print os.getcwd()
        compile_cmd = "python graphitc.py -f " + input_with_schedule_path + input_file_name + " -o test.cpp"
        print compile_cmd
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = "g++ -g -std=c++11 -I ../../src/runtime_lib/  test.cpp -o test.o"

        if use_parallel:
            print "using icpc for parallel compilation"
            cpp_compile_cmd = "icpc -g -std=c++11 -I ../../src/runtime_lib/ -DCILK test.cpp -o test.o"

        subprocess.check_call(cpp_compile_cmd, shell=True)

    def basic_compile_exec_test(self, input_file_name):
        input_with_schedule_path = '../../test/input_with_schedules/'
        print os.getcwd()
        compile_cmd = "python graphitc.py -f " + input_with_schedule_path + input_file_name + " -o test.cpp"
        print compile_cmd
        subprocess.check_call(compile_cmd, shell=True)
        subprocess.check_call("g++ -g -std=c++11 -I ../../src/runtime_lib/  test.cpp "  " -o test.o", shell=True)
        os.chdir("..")
        subprocess.check_call("bin/test.o")
        os.chdir("bin")


        # actual test cases

    def bfs_verified_test(self, input_file_name, use_separate_algo_file=False):
        if (use_separate_algo_file):
            self.basic_compile_test_with_separate_algo_schedule_files("bfs_with_filename_arg.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        os.chdir("..");
        cmd = "./bin/test.o ../test/graphs/4.el" + " > verifier_input"
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        proc = subprocess.Popen("./bin/bfs_verifier -f ../test/graphs/4.el -t verifier_input -r 8", stdout=subprocess.PIPE, shell=True)
        test_flag = False
        for line in iter(proc.stdout.readline,''):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")


    def cc_verified_test(self, input_file_name):
        self.basic_compile_test(input_file_name)
        # proc = subprocess.Popen(["./"+ self.executable_file_name], stdout=subprocess.PIPE)
        os.chdir("..")
        cmd = "./bin/test.o" + " > verifier_input"
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        proc = subprocess.Popen("./bin/cc_verifier -f ../test/graphs/4.el -t verifier_input", stdout=subprocess.PIPE, shell=True)
        test_flag = False
        for line in iter(proc.stdout.readline,''):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def sssp_verified_test(self, input_file_name, use_separate_algo_file=True):
        if (use_separate_algo_file):
            self.basic_compile_test_with_separate_algo_schedule_files("sssp.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        os.chdir("..");
        cmd = "./bin/test.o" + " > verifier_input"
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        proc = subprocess.Popen("./bin/sssp_verifier -f ../test/graphs/4.wel -t verifier_input -r 0", stdout=subprocess.PIPE, shell=True)
        test_flag = False
        for line in iter(proc.stdout.readline,''):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def pr_verified_test(self, input_file_name, use_separate_algo_file=False):
        if (use_separate_algo_file):
            self.basic_compile_test_with_separate_algo_schedule_files("pagerank_with_filename_arg.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        proc = subprocess.Popen("./"+ self.executable_file_name + " ../../test/graphs/test.el", shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        output = proc.stdout.readline()
        print "output: " + output.strip()
        self.assertEqual(float(output.strip()), 0.00289518)

    def pr_delta_verified_test(self, input_file_name):
        self.basic_compile_test(input_file_name)
        proc = subprocess.Popen("./"+ self.executable_file_name + " ../../test/graphs/test.el", shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        lines = proc.stdout.readlines()
        print lines
        self.assertEqual(float(lines[0].strip()), 1)
        # first frontier has 5 vertices
        self.assertEqual(float(lines[2].strip()), 5)
        self.assertEqual(float(lines[3].strip()),  0.566667)
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

    def cf_verified_test(self, input_file_name):
        self.basic_compile_test(input_file_name)
        proc = subprocess.Popen("./"+ self.executable_file_name + " ../../test/graphs/test_cf.wel", shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        output = proc.stdout.readline()
        print "output: " + output.strip()
        self.assertEqual(float(output.strip()), 7.49039)


    def test_simple_splitting(self):
        self.basic_compile_test("simple_loop_index_split.gt")

    def test_pagerank_AoS(self):
        #self.basic_compile_test("simple_pagerank_with_AoS.gt")
        self.basic_compile_test_with_separate_algo_schedule_files("pagerank.gt", "simple_pagerank_with_AoS.gt")

    def test_eigenvector_pagerank_fusion(self):
        self.basic_compile_test("eigenvector_pr_fusion.gt")

    def test_bfs_push_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_push_parallel_cas.gt", True)

    def test_bfs_hybrid_dense_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_hybrid_dense_parallel_cas.gt", True)

    def test_bfs_push_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_push_parallel_cas.gt", True)

    def test_bfs_pull_parallel_verified(self):
        self.bfs_verified_test("bfs_pull_parallel.gt", True)

    def test_bfs_push_sliding_queue_parallel_cas_verified(self):
        self.bfs_verified_test("bfs_push_sliding_queue_parallel_cas.gt", True)

    def test_cc_hybrid_dense_parallel_cas_verified(self):
        self.cc_verified_test("cc_hybrid_dense_parallel_cas.gt")

    def test_cc_hybrid_dense_parallel_bitvector_verified(self):
        self.cc_verified_test("cc_hybrid_dense_parallel_bitvector.gt")

    def test_cc_push_parallel_cas_verified(self):
        self.cc_verified_test("cc_push_parallel_cas.gt")


    def test_cc_pull_parallel_verified(self):
        self.cc_verified_test("cc_pull_parallel.gt")

    def test_sssp_push_parallel_cas_verified(self):
        self.sssp_verified_test("sssp_push_parallel_cas.gt")

    def test_sssp_hybrid_denseforward_parallel_cas_verified(self):
        self.sssp_verified_test("sssp_hybrid_denseforward_parallel_cas.gt")

    def test_sssp_hybrid_dense_parallel_cas_verified(self):
        self.sssp_verified_test("sssp_hybrid_dense_parallel_cas.gt")

    def test_sssp_pull_parallel_verified(self):
        self.sssp_verified_test("sssp_pull_parallel.gt")

    def test_sssp_push_parallel_sliding_queue_verified(self):
        self.sssp_verified_test("sssp_push_parallel_sliding_queue.gt")

    def test_pagerank_parallel_pull_expect(self):
        self.pr_verified_test("pagerank_pull_parallel.gt", True)

    def test_pagerank_parallel_push_expect(self):
        self.pr_verified_test("pagerank_push_parallel.gt")

    def test_pagerank_parallel_pull_load_balance_expect(self):
        self.pr_verified_test("pagerank_pull_parallel_load_balance.gt", True)

    def test_cf_parallel_expect(self):
        self.cf_verified_test("cf_pull_parallel.gt")

    def test_cf_parallel_load_balance_expect(self):
        self.cf_verified_test("cf_pull_parallel_load_balance.gt")

    def test_prdelta_parallel_pull(self):
        self.pr_delta_verified_test("pagerank_delta_pull_parallel.gt")

    def test_prdelta_parallel_load_balance_pull(self):
        self.pr_delta_verified_test("pagerank_delta_pull_parallel_load_balance.gt")

    def test_prdelta_parallel_load_balance_hybrid_dense_with_bitvec(self):
        self.pr_delta_verified_test("pagerank_delta_hybrid_dense_parallel_bitvector.gt")

    def test_prdelta_parallel_load_balance_hybrid_dense_without_bitvec(self):
        self.pr_delta_verified_test("pagerank_delta_hybrid_dense_parallel_load_balance_no_bitvector.gt")

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == "parallel":
        use_parallel = True
        print "using parallel"
        del sys.argv[1]

    unittest.main()

    # used for enabling a specific test
    # suite = unittest.TestSuite()
    # suite.addTest(TestGraphitCompiler('test_bfs_push_parallel_cas_verified'))
    # unittest.TextTestRunner(verbosity=2).run(suite)
