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

    cd graphit/test/python
    python test.py

Try Graphit
===========
To compile an example:

    cd build
    bin/graphitc -f ../test/input/simple_vector_sum.gt -o test.cpp
    g++ -std=c++11 -I ../src/runtime_lib/ test.cpp  -o test.o
    ./test.o

