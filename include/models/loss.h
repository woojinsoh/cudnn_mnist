#pragma once

#include <cuda.h>

// from softmax
__global__ void nLLoss(int batch_size, float *log_softmax, int *labels, float *loss);
__global__ void diffNLLoss(int batch_size, float *softmax_output, int *labels, float *dy);

// from log-softmax
__global__ void nLLLoss(int batch_size, float *log_softmax, int *labels, float *loss);
__global__ void diffNLLLoss(int batch_size, int *labels, float *dy);


//from (log)softmax
__global__ void calAccuracy(int batch_size, int num_classes, float *softmax, int *labels, float *accuracy);