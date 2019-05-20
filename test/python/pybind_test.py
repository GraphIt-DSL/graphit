#!/usr/local/bin/python

import unittest
import subprocess
import os
import shutil
import graphit
from scipy.sparse import csr_matrix
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
        cls.cpp_compiler = CXX_COMPILER
        cls.compile_flags = "-std=c++11"
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

    def test_pybind_pr_with_return(self):
        module = graphit.compile_and_load(self.root_test_input_dir + "export_pr_with_return.gt")
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
	

if __name__ == '__main__':
    unittest.main()
    # used for enabling a specific test

    # suite = unittest.TestSuite()
    # suite.addTest(TestGraphitCompiler('test_extern_simple_edgeset_apply'))
    # unittest.TextTestRunner(verbosity=2).run(suite)
