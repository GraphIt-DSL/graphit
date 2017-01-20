cmake_minimum_required(VERSION 3.6)
project(graphit)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories(./include/)

add_subdirectory(src)

set(SOURCE_FILES src/main.cpp include/graphit/frontend/scanner.h src/frontend/scanner.cpp include/graphit/frontend/frontend.h src/frontend/frontend.cpp include/graphit/frontend/parser.h src/frontend/parser.cpp)
add_executable(graphitc ${SOURCE_FILES})