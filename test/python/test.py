import unittest
import subprocess
import os
import shutil

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        print "build the binaries"
        if not os.path.isdir("../../build_dir"):
            #shutil.rmtree("../../build_dir")
            os.mkdir("../../build_dir")
            os.chdir("../../build_dir")
            subprocess.call(["cmake", ".."])
            subprocess.call(["make"])

    def test_main_func_no_return(self):
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)