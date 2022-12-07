cmake_minimum_required(VERSION 3.15)

project(cudnn_mnist
        DESCRIPTION "cuDNN example using MNIST dataset"
        LANGUAGES CXX CUDA)

find_package(CUDA 11.0 REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Assigning GPU architecture 
set(SM_SETS 52 60 61 70 75 80 86)
set(FIND_SM False)
if(NOT SM)
    message(FATAL_ERROR "Please assign your GPU architecture(i.e.,SM should be one of 52,60,61,70,75,80,86)")
endif()
if (SM)
   foreach(SM_NUM IN LISTS SM_SETS)
       string(FIND "${SM}" "${SM_NUM}" SM_POS)
       if (SM_POS GREATER -1)
           set(CMAKE_CUDA_ARCHITECTURES ${SM_NUM})  # it should be properly set. if not, unexpected results should occur when you execute the binary. e.g., A100: SM=80, V100: SM=70   
           set(FIND_SM True)
           message(STATUS "Assigned GPU architecture: ${SM_NUM}")
       endif()
    endforeach()
endif()
if (NOT FIND_SM)
    message(FATAL_ERROR "Wrong SM.Please assign the right one(i.e.,SM should be one of 52,60,61,70,75,80,86)")
endif()

# Compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Xcompiler -Wall -ldl")
add_subdirectory(src)