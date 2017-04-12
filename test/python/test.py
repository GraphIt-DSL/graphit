#!/usr/local/bin/python

import unittest
import subprocess
import os
import shutil

class TestGraphitCompiler(unittest.TestCase):
    first_time_setup = True

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir("../../build_dir"):
            print "build the binaries"
            #shutil.rmtree("../../build_dir")
            os.mkdir("../../build_dir")
            os.chdir("../../build_dir")
            subprocess.call(["cmake", ".."])
            subprocess.call(["make"])
        else:
            os.chdir("../../build_dir")

        cls.root_test_input_dir = "../test/input/"
        cls.cpp_compiler = "g++"
        cls.output_file_name = "test.cpp"
        cls.executable_file_name = "test.o"

    def setUp(self):
        self.clean_up()

    def tearDown(self):
        self.clean_up()

    def clean_up(self):
        #clean up previously generated files
        if os.path.isfile(self.output_file_name):
            os.remove(self.output_file_name)
        if os.path.isfile(self.executable_file_name):
            os.remove(self.executable_file_name)

    def basic_compile_test(self, input_file_name):

        # "-f" and "-o" must not have space in the string, otherwise it doesn't read correctly
        graphit_compile_cmd = ["bin/graphitc", "-f", self.root_test_input_dir + input_file_name, "-o" , self.output_file_name]
        # check the return code of the call as a way to check if compilation happened correctly
        self.assertEqual(subprocess.call(graphit_compile_cmd), 0)
        # check if g++ compilation succeeded
        self.assertEqual(
            subprocess.call([self.cpp_compiler, self.output_file_name, "-o", self.executable_file_name]),
            0)
        self.assertEqual(subprocess.call(["./"+ self.executable_file_name]), 0)

    # expect to fail graphit compiler
    def basic_compile_test_graphit_compile_fail(self, input_file_name):
        graphit_compile_cmd = ["bin/graphitc", "-f", self.root_test_input_dir + input_file_name, "-o" , self.output_file_name]
        self.assertNotEqual(subprocess.call(graphit_compile_cmd), 0)

    # expect to fail c++ compiler (but will NOT fail graphit compiler)
    def basic_compile_test_cpp_compile_fail(self, input_file_name):
        graphit_compile_cmd = ["bin/graphitc", "-f", self.root_test_input_dir + input_file_name, "-o" , self.output_file_name]
        self.assertEqual(subprocess.call(graphit_compile_cmd), 0)
        self.assertNotEqual(
            subprocess.call([self.cpp_compiler, self.output_file_name, "-o", self.executable_file_name]),
            0)


    def test_main(self):
        self.basic_compile_test("simple_main.gt")

    def test_main_fail(self):
        self.basic_compile_test_graphit_compile_fail("simple_main_fail.gt")

    def test_main_print_four(self):
        self.basic_compile_test("simple_main.gt")
        proc = subprocess.Popen(["./"+ self.executable_file_name], stdout=subprocess.PIPE)
        #check the value printed to stdout is 4
        output = proc.stdout.readline()
        self.assertEqual(int(output.strip()), 4)

    def test_simple_func_add_no_return(self):
        self.basic_compile_test_cpp_compile_fail("simple_func_add_no_return.gt")

if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphitCompiler)
    unittest.TextTestRunner(verbosity=2).run(suite)