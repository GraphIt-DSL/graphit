cmake_minimum_required(VERSION 3.5)
project(graphit)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# build the compiler
include_directories(./include/)
include_directories(./src/runtime_lib)
add_subdirectory(src)

# header files
#file(GLOB HEADER_FILES include/graphit/frontend/*.h include/graphit/midend/*.h include/graphit/backend/*.h include/graphit/utils/*.h)
# file(GLOB_RECURSE HEADER_FILES include/*.h src/runtime_lib/*.h)
file(GLOB_RECURSE HEADER_FILES include/*.h)


file(GLOB_RECURSE SOURCE_FILES include/*.h include/graphit/midend/*.h src/*.cpp src/frontend/*.cpp src/midend/*.cpp src/backend/*.cpp src/utils/*.cpp src/utils/*.h include/utils/*.h)
add_executable(graphitc ${SOURCE_FILES} src/default_schedule.cpp test/library_test_drivers/library_test_driver_cpp.txt)

# build a front end library used for unit testing
file(GLOB_RECURSE LIB_SOURCE_FILES include/*.h src/frontend/*.cpp src/midend/*.cpp src/backend/*.cpp src/utils/*.cpp src/main.cpp)

add_library(graphitlib ${HEADER_FILES} ${LIB_SOURCE_FILES})

# build test suite
include_directories(./test/gtest/)
add_subdirectory(test)
set(GTEST_SOURCE ./test/gtest/gtest-all.cc)
set(GTEST_HEADER ./test/gtest/gtest.h test/input/extern_add_one.cpp src/runtime_lib/infra_gapbs/minimum_spanning_tree.h)
add_library(gtest ${GTEST_HEADER} ${GTEST_SOURCE})


file(GLOB TEST_SOURCE_FILES test/c++/*.cpp)
add_executable(graphit_test ${TEST_SOURCE_FILES})
target_link_libraries(graphit_test gtest)
target_link_libraries(graphit_test pthread)

target_link_libraries(graphit_test graphitlib)

# Python tests
set(PYTHON_SOURCE_FILES ./test/python/test.py)
#add_executable(PYTHON_TEST ${PYTHON_SOURCE_FILES})
#target_link_libraries(PYTHON_TEST ${PYTHON_LIBRARIES})

#configure_file(src/graphitc.py ${CMAKE_BINARY_DIR}/bin/graphitc.py COPYONLY)
set(GRAPHITC_PY
	"${CMAKE_BINARY_DIR}/bin/graphitc.py"
)
add_custom_command(OUTPUT ${GRAPHITC_PY}
	COMMAND sed -e s?\$\{CXX_COMPILER\}?${CMAKE_CXX_COMPILER}?g -e s?\$\{GRAPHIT_SOURCE_DIRECTORY\}?${CMAKE_SOURCE_DIR}?g -e s?\$\{GRAPHIT_BUILD_DIRECTORY\}?${CMAKE_BINARY_DIR}?g ${CMAKE_SOURCE_DIR}/src/graphitc.py > ${GRAPHITC_PY}
	DEPENDS ${CMAKE_SOURCE_DIR}/src/graphitc.py
	VERBATIM
)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/python_tests/test_with_schedules.py
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/python_tests
	COMMAND sed -e s?\$\{CXX_COMPILER\}?${CMAKE_CXX_COMPILER}?g -e s?\$\{GRAPHIT_SOURCE_DIRECTORY\}?${CMAKE_SOURCE_DIR}?g -e s?\$\{GRAPHIT_BUILD_DIRECTORY\}?${CMAKE_BINARY_DIR}?g ${CMAKE_SOURCE_DIR}/test/python/test_with_schedules.py > ${CMAKE_BINARY_DIR}/python_tests/test_with_schedules.py
	DEPENDS ${CMAKE_SOURCE_DIR}/test/python/test_with_schedules.py
	VERBATIM
)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/python_tests/test.py
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/python_tests
	COMMAND sed -e s?\$\{CXX_COMPILER\}?${CMAKE_CXX_COMPILER}?g -e s?\$\{GRAPHIT_SOURCE_DIRECTORY\}?${CMAKE_SOURCE_DIR}?g -e s?\$\{GRAPHIT_BUILD_DIRECTORY\}?${CMAKE_BINARY_DIR}?g ${CMAKE_SOURCE_DIR}/test/python/test.py > ${CMAKE_BINARY_DIR}/python_tests/test.py
	DEPENDS ${CMAKE_SOURCE_DIR}/test/python/test.py
	VERBATIM
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/python_tests/pybind_test.py
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/python_tests
	COMMAND sed -e s?\$\{CXX_COMPILER\}?${CMAKE_CXX_COMPILER}?g -e s?\$\{GRAPHIT_SOURCE_DIRECTORY\}?${CMAKE_SOURCE_DIR}?g -e s?\$\{GRAPHIT_BUILD_DIRECTORY\}?${CMAKE_BINARY_DIR}?g ${CMAKE_SOURCE_DIR}/test/python/pybind_test.py > ${CMAKE_BINARY_DIR}/python_tests/pybind_test.py
	DEPENDS ${CMAKE_SOURCE_DIR}/test/python/pybind_test.py
	VERBATIM
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/graphit.py 
	COMMAND sed -e s?\$\{GRAPHIT_SOURCE_DIRECTORY\}?${CMAKE_SOURCE_DIR}?g -e s?\$\{CXX_COMPILER\}?${CMAKE_CXX_COMPILER}?g -e s?\$\{GRAPHIT_BUILD_DIRECTORY\}?${CMAKE_BINARY_DIR}?g ${CMAKE_SOURCE_DIR}/src/python/graphit.py > ${CMAKE_BINARY_DIR}/graphit.py
	DEPENDS ${CMAKE_SOURCE_DIR}/src/python/graphit.py
	VERBATIM
)

add_custom_target(copy_graphitc_py ALL DEPENDS ${GRAPHITC_PY})
add_custom_target(copy_python_tests ALL DEPENDS ${CMAKE_BINARY_DIR}/python_tests/test_with_schedules.py ${CMAKE_BINARY_DIR}/python_tests/test.py ${CMAKE_BINARY_DIR}/python_tests/pybind_test.py)
add_custom_target(copy_graphit_py ALL DEPENDS ${CMAKE_BINARY_DIR}/graphit.py)

configure_file(src/main.cpp ${CMAKE_BINARY_DIR}/bin/main.cpp COPYONLY)
configure_file(test/library_test_drivers/library_test_driver_cpp.txt ${CMAKE_BINARY_DIR}/bin/library_test_driver.cpp COPYONLY)
configure_file(test/library_test_drivers/library_test_driver_weighted_cpp.txt ${CMAKE_BINARY_DIR}/bin/library_test_driver_weighted.cpp COPYONLY)
configure_file(test/library_test_drivers/library_test_driver_cf_cpp.txt ${CMAKE_BINARY_DIR}/bin/library_test_driver_cf.cpp COPYONLY)


add_executable(bc_verifier ./test/verifiers/bc_verifier.cpp)
add_executable(bfs_verifier ./test/verifiers/bfs_verifier.cpp)
add_executable(sssp_verifier ./test/verifiers/sssp_verifier.cpp)
add_executable(tc_verifier ./test/verifiers/tc_verifier.cpp)
add_executable(ppsp_verifier ./test/verifiers/ppsp_verifier.cpp)
add_executable(cc_verifier ./test/verifiers/cc_verifier.cpp)
