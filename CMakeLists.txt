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
file(GLOB SOURCE_FILES src/*.cpp src/frontend/*.cpp src/midend/*.cpp src/backend/*.cpp src/utils/*.cpp src/utils/*.h)
add_executable(graphitc ${SOURCE_FILES})

# build a front end library used for unit testing
file(GLOB LIB_SOURCE_FILES src/frontend/*.cpp src/midend/*.cpp src/backend/*.cpp src/utils/*.cpp src/utils/*.h)
file(GLOB HEADER_FILES include/frontend/*.h include/midend/*.h include/backend/*.h include/utils/*.h)

add_library(graphitlib ${HEADER_FILES} ${LIB_SOURCE_FILES})

# build test suite
include_directories(./test/gtest/)
add_subdirectory(test)
set(GTEST_SOURCE ./test/gtest/gtest-all.cc)
set(GTEST_HEADER ./test/gtest/gtest.h)
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})


file(GLOB TEST_SOURCE_FILES test/c++/*.cpp)
add_executable(graphit_test ${TEST_SOURCE_FILES})
target_link_libraries(graphit_test gtest)
target_link_libraries(graphit_test graphitlib)
