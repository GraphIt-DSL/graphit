#!/usr/local/bin/python

import unittest
import subprocess
import os
import shutil
import sys

use_parallel = False
use_numa = False

GRAPHIT_BUILD_DIRECTORY="${GRAPHIT_BUILD_DIRECTORY}".strip().rstrip("/")
GRAPHIT_SOURCE_DIRECTORY="${GRAPHIT_SOURCE_DIRECTORY}".strip().rstrip("/")
CXX_COMPILER="${CXX_COMPILER}"


class TestGraphitCompiler(unittest.TestCase):
    first_time_setup = True

    #NOTE: currently the tests can only work within the build/bin directory
    @classmethod
    def setUpClass(cls):
        build_dir = GRAPHIT_BUILD_DIRECTORY
        if not os.path.isdir(build_dir):
	    # This can never be true now since the test is run from the build directory
            print ("build the binaries")
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
        cls.root_test_input_dir = GRAPHIT_SOURCE_DIRECTORY + "/test/input/"
        cls.compile_flags = "-std=c++11"
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
        input_algos_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input/'
        input_schedules_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input_with_schedules/'
        print ("current directory: " + os.getcwd())
        algo_file = input_algos_path + input_algo_file
        schedule_file = input_schedules_path + input_schedule_file
        compile_cmd = "python graphitc.py -a " + algo_file + " -f " + schedule_file + " -o test.cpp"
        print (compile_cmd)
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = self.cpp_compiler + " -g -std=c++11 -I "+self.include_path+ " " + self.numa_flags + " test.cpp -o test.o"
        if use_parallel:
            print ("using icpc for parallel compilation")
            cpp_compile_cmd = "icpc -g -std=c++11 -I " + self.include_path +" "+ self.parallel_framework + " " + self.numa_flags + " test.cpp -o test.o"
        print (cpp_compile_cmd)
        subprocess.check_call(cpp_compile_cmd, shell=True)

    # compiles the input file with both the algorithm and schedule specification
    def basic_compile_test(self, input_file_name):
        input_with_schedule_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input_with_schedules/'
        print ("current directory: " + os.getcwd())
        compile_cmd = "python graphitc.py -f " + input_with_schedule_path + input_file_name + " -o test.cpp"
        print (compile_cmd)
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = self.cpp_compiler + " -g -std=c++11 -I "+self.include_path+" " + self.numa_flags + " test.cpp -o test.o"

        if use_parallel:
            print ("using icpc for parallel compilation")
            cpp_compile_cmd = "icpc -g -std=c++11 -I "+self.include_path+" " + self.parallel_framework + " " + self.numa_flags + " test.cpp -o test.o"

        subprocess.check_call(cpp_compile_cmd, shell=True)

    def basic_compile_exec_test(self, input_file_name):
        input_with_schedule_path = GRAPHIT_SOURCE_DIRECTORY + '/test/input_with_schedules/'
        print ("current directory: " + os.getcwd())
        compile_cmd = "python graphitc.py -f " + input_with_schedule_path + input_file_name + " -o test.cpp"
        print (compile_cmd)
        subprocess.check_call(compile_cmd, shell=True)
        cpp_compile_cmd = self.cpp_compiler + " -g -std=c++11 -I "+self.include_path+" " + self.numa_flags + " test.cpp -o test.o"
        subprocess.check_call(cpp_compile_cmd, shell=True)
        os.chdir("..")
        subprocess.check_call("bin/test.o")
        os.chdir("bin")


        # actual test cases

    def bfs_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bfs_with_filename_arg.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        os.chdir("..");
        cmd = "OMP_PLACES=sockets ./bin/test.o "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/4.el" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/bfs_verifier -f "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/4.el -t verifier_input -r 8"
        print (verify_cmd)
        proc = subprocess.Popen(verify_cmd, stdout=subprocess.PIPE, shell=True)
        test_flag = False
        for line in iter(proc.stdout.readline,''):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")


    def cc_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("cc.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)        # proc = subprocess.Popen(["./"+ self.executable_file_name], stdout=subprocess.PIPE)
        os.chdir("..")
        cmd = "OMP_PLACES=sockets ./bin/test.o" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/cc_verifier -f "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/4.el -t verifier_input"
        print (verify_cmd)
        proc = subprocess.Popen(verify_cmd, stdout=subprocess.PIPE, shell=True)
        test_flag = False
        for line in iter(proc.stdout.readline,''):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def bc_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bc.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)        # proc = subprocess.Popen(["./"+ self.executable_file_name], stdout=subprocess.PIPE)
        os.chdir("..")
        cmd = "OMP_PLACES=sockets ./bin/test.o "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/4.el" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/bc_verifier -f ../test/graphs/4.el -t verifier_input -r 3"
        print (verify_cmd)
        proc = subprocess.Popen(verify_cmd, stdout=subprocess.PIPE, shell=True)
        test_flag = False
        for line in iter(proc.stdout.readline,''):
            if line.rstrip().find("SUCCESSFUL") != -1:
                test_flag = True
                break;
        self.assertEqual(test_flag, True)
        os.chdir("bin")

    def sssp_verified_test(self, input_file_name, use_separate_algo_file=True):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("sssp.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        os.chdir("..");
        cmd = "OMP_PLACES=sockets ./bin/test.o" + " > verifier_input"
        print (cmd)
        subprocess.call(cmd, shell=True)

        # invoke the BFS verifier
        verify_cmd = "./bin/sssp_verifier -f "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/4.wel -t verifier_input -r 0"
        print (verify_cmd)
        proc = subprocess.Popen(verify_cmd, stdout=subprocess.PIPE, shell=True)
        test_flag = False
        for line in iter(proc.stdout.readline,''):
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
            cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test.el"
        else:
            cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test.el  2"
        print (cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        output = proc.stdout.readline()
        print ("output: " + output.strip())
        self.assertEqual(float(output.strip()), 0.00289518)

    def bc_basic_compile_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("bc.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test.el"

    def pr_delta_verified_test(self, input_file_name, use_separate_algo_file=False):
        if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("pr_delta.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test.el"
        print (cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        lines = proc.stdout.readlines()
        print (lines)
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

    def cf_verified_test(self, input_file_name, use_separate_algo_file=False):
        if (use_separate_algo_file):
            self.basic_compile_test_with_separate_algo_schedule_files("cf.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test_cf.wel"
        print (cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        output = proc.stdout.readline()
        print ("output: " + output.strip())
        self.assertEqual(float(output.strip()), 7.49039)

    def eigenvector_centrality_verified_test(self, input_file_name, use_separate_algo_file=False):
	if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("eigenvector_centrality.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test.el"
        print (cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        lines = proc.stdout.readlines()
        print (lines)
        self.assertEqual(float(lines[0].strip()), 3.2)

    def closeness_centrality_unweighted_test(self, input_file_name, use_separate_algo_file=False):
	if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("closeness_centrality_unweighted.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test.el"
        print (cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        lines = proc.stdout.readlines()
        print (lines)
        self.assertEqual(float(lines[3].strip()), 3)

    def closeness_centrality_weighted_test(self, input_file_name, use_separate_algo_file=False):
	if use_separate_algo_file:
            self.basic_compile_test_with_separate_algo_schedule_files("closeness_centrality_weighted.gt", input_file_name)
        else:
            self.basic_compile_test(input_file_name)
        cmd = "OMP_PLACES=sockets ./"+ self.executable_file_name + " "+GRAPHIT_SOURCE_DIRECTORY+"/test/graphs/test.el"
        print (cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        #check the value printed to stdout is as expected
        lines = proc.stdout.readlines()
        print (lines)
        self.assertEqual(float(lines[1].strip()), 15)


    def test_simple_splitting(self):
        self.basic_compile_test("simple_loop_index_split.gt")

    def test_pagerank_AoS(self):
        #self.basic_compile_test("simple_pagerank_with_AoS.gt")
        self.basic_compile_test_with_separate_algo_schedule_files("pagerank.gt", "simple_pagerank_with_AoS.gt")

    def test_filter_sum_parallel(self):
        self.basic_compile_test_with_separate_algo_schedule_files("simple_apply_sum.gt", "simple_vector_sum.gt")

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
        self.closeness_centrality_unweighted_test("closeness_centrality_unweighted_hybrid_parallel.gt",True)	
    
    def test_closeness_centrality_weighted_hybrid_parallel(self):
        self.closeness_centrality_weighted_test("closeness_centrality_weighted_hybrid_parallel.gt",True)	

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



if __name__ == '__main__':
    while len(sys.argv) > 1:
        if "parallel" in sys.argv:
            use_parallel = True
            print ("using parallel")
            del sys.argv[sys.argv.index("parallel")]
        if "numa" in sys.argv:
            use_numa = True
            print ("using numa")
            del sys.argv[sys.argv.index("numa")]
    
    unittest.main()

    #used for enabling a specific test

    # suite = unittest.TestSuite()
    # suite.addTest(TestGraphitCompiler('test_bc_SparsePush_verified'))
    # unittest.TextTestRunner(verbosity=2).run(suite)


