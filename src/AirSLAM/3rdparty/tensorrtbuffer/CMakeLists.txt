cmake_minimum_required(VERSION 3.6)
project(tensorrtbuffer)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

include_directories("/home/bhp/work_bhp/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8/TensorRT-8.6.1.6/include/")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(tensorrtbuffer SHARED src/logger.cpp src/sample_options.cpp src/sample_utils.cpp)



