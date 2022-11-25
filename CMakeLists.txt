cmake_minimum_required(VERSION 3.15)

project(cudnn_mnist
        DESCRIPTION "cuDNN example using MNIST dataset"
        LANGUAGES CXX CUDA)

find_package(CUDA 11.0 REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_ARCHITECTURES 80) # Ampere A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Xcompiler -Wall -ldl")

add_subdirectory(src)

