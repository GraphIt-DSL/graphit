 # graphit
GraphIt DSL compiler.

Build Graphit
===========
To perform an out-of-tree build of Graphit do:

After you have cloned the directory:

    cd graphit
    mkdir build
    cd build
    cmake ..
    make


To run the C++ test suite do (all tests should pass):

    cd build/bin
    ./graphit_test

To run the Python end-to-end test suite:

start at the top level graphit directory cloned from Github, NOT the build directory
(All tests would pass, but some would generate error messages from the g++ compiler)
Currently the project supports Python 2.x and not Python 3.x (the print syntax is different)

    cd graphit/test/python
    python test.py
    python test_with_schedules.py

Try Graphit
===========
GraphIt compiler currently generates a C++ output file from the .gt input GraphIt programs. 
To compile an input file with schedules in the same file (assuming the build directory is in the root project directory)

    cd build/bin
    python graphitc.py -f ../../test/input/simple_vector_sum.gt -o test.cpp
    g++ -std=c++11 -I ../../src/runtime_lib/ test.cpp  -o test.o
    ./test.o

To compile an input algorithm file and another separate schedule file (some of the test files have hardcoded paths to test inputs, be sure to modify that or change the directory you run the compiled files)

```
   cd build/bin
   python graphitc.py -a ../../test/input/cc.gt -f ../../test/input_with_schedules/cc_pull_parallel.gt -o test.cpp
   g++ -std=c++11 -I ../../src/runtime_lib/ test.cpp  -o test.o
   ./test.o 
```