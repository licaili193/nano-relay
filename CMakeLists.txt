
cmake_minimum_required (VERSION 3.10.2)
project (Relay)

set(CMAKE_CXX_STANDARD 14)
message("CXX Standard: ${CMAKE_CXX_STANDARD}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(CUDA REQUIRED)
find_package(gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(OpenGL REQUIRED)

set (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

add_subdirectory (src)