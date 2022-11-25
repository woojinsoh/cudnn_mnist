#pragma once

#include <cudnn.h>
#include <cuda.h>
#include <cublas_v2.h>

void checkCUDNN(cudnnStatus_t status);

void checkCUDA(cudaError_t error);

void checkCUBLAS(cublasStatus_t status);

void oneHotEncoding(int batch_size, int num_classes, int *labels, int *onehot_labels);

/* used to extend bias(1x_out_features) to (batch_size x _out_feature) */
__global__ void getOneVec(float *vec, size_t length);

__global__ void takeExponential(float *vec);

/* Tensor Transformation between NCHW and NHWC */
__global__ void permute3dTensorsKernel(float *input, int64_t *input_dims, int64_t *sp, float *output);

__global__ void permute3dTensorsKernel_v2(float *input, int64_t *input_dims, int64_t *sp, float *output);

__device__ int64_t tensorIndexToFlat(int64_t *tensor_index, int64_t *tensor_dims);

__device__ void flatToTensorIndex(int64_t flat_index, int64_t *tensor_index, int64_t *tensor_dims);
