#include <iostream>
#include <malloc.h>
#include "utils.h"

//* CUDA Errors *//
void checkCUDNN(cudnnStatus_t status){    
    if (status != CUDNN_STATUS_SUCCESS){
        printf("CUDNN ERROR %d:: %s\n", status, cudnnGetErrorString(status));
        exit(-1);
    }
}

void checkCUDA(cudaError_t error){
    if(error != cudaSuccess){
        printf("CUDA ERROR %d:: %s\n", error, cudaGetErrorString(error));
        // std::cout << "CUDA ERROR:: " << error << std::endl;
    }
}

void checkCUBLAS(cublasStatus_t status){
    if(status != CUBLAS_STATUS_SUCCESS){
        std::cout << "CUBLAS ERROR:: " << status << std::endl;                                                 
    }
}
/*******************/


void oneHotEncoding(int batch_size, int num_classes, int *labels, int *onehot_labels)
// labels = [batch_size]
// onehot_labels = [batch_size * num_classes]
{
    for(int i=0; i<batch_size; i++){
        for(int j=0; j<num_classes; j++){
            if(labels[i] == j){
                onehot_labels[i * num_classes + j] = 1;
            } 
            else{
                onehot_labels[i * num_classes + j] = 0;
            }
        }
    }
}

__global__ void getOneVec(float* vec, size_t length){
    // used to extend bias(1 x _out_features) to (batch_size x _out_feature) 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length){
        vec[idx] = 1.f;
    }
}

__global__ void takeExponential(float *vec){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    vec[idx] = expf(vec[idx]);
}

// permute 3d tesnors such as [N, C*H, W] to [N, W, C*H]
__global__ void permute3dTensorsKernel(float *input, int64_t *input_dims, int64_t *sp, float *output) {
    int64_t output_index;
    int64_t input_tensor_index[3];
    int64_t output_tensor_index[3];
    int64_t output_dims[3];
    
    #pragma unroll
    for(int i = 0; i < 3; i++){
        output_dims[sp[i]] = input_dims[i];
    }
    int64_t input_index = blockDim.x * blockIdx.x + threadIdx.x;
            
    if(input_index < input_dims[0] * input_dims[1] * input_dims[2]){     
        // Flat Index to Tensor Index
        int64_t tmp_index = input_index;
        for (int i = 2; i >= 0; i--) {
            int64_t new_index = tmp_index / input_dims[i];
            input_tensor_index[i] = tmp_index - input_dims[i] * new_index;
            tmp_index = new_index;
        }

        // Align permuted index
        #pragma unroll
        for (int i = 0; i < 3; i++){
            output_tensor_index[i] = input_tensor_index[sp[i]];
        }
        
        // Tensor Index to Flat Index
        output_index = output_tensor_index[0];   //batch_dim
        for (int i = 1; i < 3; i++) {
            output_index = output_index * output_dims[i] + output_tensor_index[i];
        }

        // Swap
        output[output_index] = input[input_index];
    }
}

// permute 3d tesnors such as [N, C*H, W] to [N, W, C*H]
__global__ void permute3dTensorsKernel_v2(float *input, int64_t *input_dims, int64_t *sp, float *output) {
    int64_t* input_tensor_index = new int64_t[3];
    int64_t* output_tensor_index = new int64_t[3];
    int64_t* output_dims = new int64_t[3];
    int64_t output_flat_index;

    output_dims[sp[0]] = input_dims[0];
    output_dims[sp[1]] = input_dims[1];
    output_dims[sp[2]] = input_dims[2];
    
    int input_flat_index = blockDim.x * blockIdx.x + threadIdx.x;

    if(input_flat_index < input_dims[0] * input_dims[1] * input_dims[2]){        
        flatToTensorIndex(input_flat_index, input_tensor_index, input_dims);
    
        // Align permuted index
        for (int i = 0; i < 3; i++){
            output_tensor_index[i] = input_tensor_index[sp[i]];
        }
        
        output_flat_index = tensorIndexToFlat(output_tensor_index, output_dims);
        output[output_flat_index] = input[input_flat_index];
    }

    delete[] input_tensor_index;
    delete[] output_tensor_index;
    delete[] output_dims;
}

// e.g., let tensor_dim=(2,3,4) then, tensor_indice are (0,0,0), (0,0,1), ...(0,0,3), (0,1,0), ...
__device__ int64_t tensorIndexToFlat(int64_t *tensor_index, int64_t *tensor_dims) {
  int64_t flat_index = tensor_index[0];   //batch_dim
  for (int64_t i = 1; i < 3; i++) {
    flat_index = flat_index * tensor_dims[i] + tensor_index[i];
  }
  return flat_index;
}

__device__ void flatToTensorIndex(int64_t flat_index, int64_t* tensor_index, int64_t *tensor_dims) {
//   int64_t* tensor_index = new int64_t[3];
  for (int64_t i = 2; i >= 0; i--) {
    int64_t new_index = flat_index / tensor_dims[i];
    tensor_index[i] = flat_index - tensor_dims[i] * new_index;
    flat_index = new_index;
  }
//   return tensor_index;
}
