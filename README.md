# graphit
GraphIt DSL compiler.



How to compile an example

goto the build directory

bin/graphitc -f ../test/input/simple_vector_sum.gt -o test.cpp
g++ -std=c++11 -I ../src/runtime_lib/ test.cpp  -o test.o

