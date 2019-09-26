#!/usr/local/bin/python

import unittest
import subprocess
import os
import shutil
import graphit
from scipy.sparse import csr_matrix
import scipy.io
import numpy as np

GRAPHIT_BUILD_DIRECTORY="${GRAPHIT_BUILD_DIRECTORY}".strip().rstrip("/")
GRAPHIT_SOURCE_DIRECTORY="${GRAPHIT_SOURCE_DIRECTORY}".strip().rstrip("/")
CXX_COMPILER="${CXX_COMPILER}"


class TestGraphitCompiler(unittest.TestCase):
    first_time_setup = True

    @classmethod
    def setUpClass(cls):
        build_dir = GRAPHIT_BUILD_DIRECTORY
        if not os.path.isdir(build_dir):
	    #This should never be true because test is invoked from build not source
            print ("build the binaries")
            #shutil.rmtree("../../build_dir")
            os.mkdir(build_dir)
            os.chdir(build_dir)
            subprocess.call(["cmake", ".."])
            subprocess.call(["make"])
        else:
            os.chdir(build_dir)

        cwd = os.getcwd()

        cls.root_test_input_dir = GRAPHIT_SOURCE_DIRECTORY + "/test/input/"
        cls.root_test_graph_dir = GRAPHIT_SOURCE_DIRECTORY + "/test/graphs/"
        cls.cpp_compiler = CXX_COMPILER
        cls.compile_flags = "-std=c++14"
        cls.include_path = GRAPHIT_SOURCE_DIRECTORY + "/src/runtime_lib/"
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

    # utilities for setting up tests
    # csr_matrix(([4, 5, 6, 4, 5, 6], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]) is a graph with 4 vertices

    def test_pybind_pr_with_return(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_pr_with_return.gt")
        graph = csr_matrix(([0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        ranks = module.export_func(graph)
        self.assertEqual(np.sum(ranks), 1.0)	
    
    # make sure compile_and_load_cache function still gives correct answer
    def test_pybind_pr_with_cache_return(self):
        module = graphit.compile_and_load_cache(self.root_test_input_dir + "export_pr_with_return.gt")
        graph = csr_matrix(([0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        ranks = module.export_func(graph)
        self.assertEqual(np.sum(ranks), 1.0)	
    
    def test_pybind_sssp(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_sssp.gt")
        graph = csr_matrix(([4, 5, 6, 4, 5, 6], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        distances = module.export_func(graph)
        self.assertEqual(len(distances), 4)
        self.assertEqual(distances[0], 0)
        self.assertEqual(distances[1], 4)
        self.assertEqual(distances[2], 5)
        self.assertEqual(distances[3], 6)

    def test_pybind_sssp_UW(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_sssp_UW.gt")
        graph = csr_matrix(([4, 5, 6, 4, 5, 6], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        distances = module.do_sssp(graph, 0)
        self.assertEqual(len(distances), 4)
        self.assertEqual(distances[0], 0)
        self.assertEqual(distances[1], 4)
        self.assertEqual(distances[2], 5)
        self.assertEqual(distances[3], 6)

    def test_pybind_extern_apply_src_add_one(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_extern_simple_edgeset_appply.gt", [self.root_test_input_dir + "extern_src_add_one.cpp"])
        graph = csr_matrix(([4, 5, 6, 4, 5, 6], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        sum_returned = module.export_func(graph)
        self.assertEqual(sum_returned, 6)
    
    def test_pybind_pr_delta(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_pr_delta.gt")
        graph = csr_matrix(([4, 5, 6, 4, 5, 6], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        ranks = module.export_func(graph)
        self.assertEqual(len(ranks), 4)
        self.assertTrue(abs(np.sum(ranks)-1.0) < 0.001)

    def test_pybind_pr_delta_UW(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_pagerank_delta.gt")
        graph = csr_matrix(([4, 5, 6, 4, 5, 6], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        module.set_graph(graph)
        ranks = module.do_pagerank_delta()
        self.assertEqual(len(ranks), 4)
        self.assertTrue(abs(np.sum(ranks)-1.0) < 0.001)

    def test_pybind_pr_with_vector_input(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_pagerank_with_vector_input.gt")
        graph = csr_matrix(([0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        new_rank_array = np.array([0, 0, 0, 0, 0], dtype = np.int32)
        ranks = module.export_func(graph, new_rank_array)
        self.assertEqual(np.sum(ranks), 1.0)

    def test_pybind_pr_load_file(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_pr_with_return.gt")
        graph = csr_matrix(scipy.io.mmread(self.root_test_graph_dir+"4.mtx"))
        ranks = module.export_func(graph)
        self.assertEqual(len(ranks), graph.shape[0])
        self.assertTrue(abs(np.sum(ranks)-1.0) < 0.1)
    
    def test_pybind_cf(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_cf_vector_input_with_return.gt")
        graph = csr_matrix(([0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        cf_result = module.export_func(graph)
        self.assertEqual(cf_result.shape, (4, 1))
        
    def test_pybind_vector_of_vector_arg(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_vector_of_vector.gt")
        graph = csr_matrix(([0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        vector_of_vector = np.ones((4, 100))
        output_data = module.export_func(graph, vector_of_vector)
        self.assertEqual(output_data.sum(), 404)

    def test_pybind_vector_of_constant_size_arg(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_vector_of_constant_size_arg.gt")
        graph = csr_matrix(([0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        vector_of_constant_size = np.ones(100)
        module.export_func(graph, vector_of_constant_size)

    def test_pybind_various_type_vector_args(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_various_types_vector_arg.gt")
        graph = csr_matrix(([0, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0], [0, 3, 4, 5, 6]))
        vector_of_vector = np.ones((4, 100))
        vector_of_constant_size = np.ones(100)
        output = module.export_func(graph, vector_of_vector, vector_of_constant_size)
        self.assertEqual(output.sum(), 800)
   
    def test_pybind_constant_size_vector_return(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_constant_size_vector_return.gt")
        vector_return = module.export_func()
        self.assertEqual(len(vector_return), 10)
        self.assertEqual(np.sum(vector_return), 55)

    def test_pybind_constant_size_vector_of_vector_return(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_constant_size_vector_of_vector_return.gt")
        vector_return = module.export_func()
        self.assertEqual(vector_return.shape, (10, 10))
        self.assertEqual(np.sum(vector_return), 550)

if __name__ == '__main__':
    unittest.main()
    # used for enabling a specific test

    # suite = unittest.TestSuite()
    # suite.addTest(TestGraphitCompiler('test_pybind_various_type_vector_args'))
    # unittest.TextTestRunner(verbosity=2).run(suite)
