cmake_minimum_required(VERSION 3.6)
project(graphit)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# build the compiler
include_directories(./include/)
add_subdirectory(src)
set(SOURCE_FILES src/main.cpp include/graphit/frontend/scanner.h src/frontend/scanner.cpp include/graphit/frontend/frontend.h src/frontend/frontend.cpp include/graphit/frontend/parser.h src/frontend/parser.cpp)
add_executable(graphitc ${SOURCE_FILES})

# build test suite
include_directories(./test/gtest/)
add_subdirectory(test)
set(GTEST_SOURCE ./test/gtest/gtest-all.cc)
set(GTEST_HEADER ./test/gtest/gtest.h)
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})


set(TEST_SOURCE_FILES ./test/c++/frontend_test.cpp)
add_executable(graphit_test ${TEST_SOURCE_FILES})
target_link_libraries(graphit_test gtest)
