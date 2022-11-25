#include "loss.h"

/////////////////////////////////////
//* CrossEntropyLoss from softmax *//
/////////////////////////////////////
__global__ void nLLoss(int batch_size, float *softmax, int *labels, float *loss)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Negative likelihood 
    if(labels[idx]==1){
        atomicAdd(loss, (float)(logf(softmax[idx]) * -1 / batch_size));
    }    
}

__global__ void diffNLLoss(int batch_size, float *softmax_output, int *labels, float *dy)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    labels[idx]==1 ? dy[idx] = (float)(-1.f / softmax_output[idx] / batch_size) : 0.f;
}



/////////////////////////////////////
//* CrossEntropy from log_softmax *//
/////////////////////////////////////
__global__ void nLLLoss(int batch_size, float *log_softmax, int *labels,  float *loss)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Negative log-likelihood   
    if(labels[idx]==1){
        atomicAdd(loss, (float)(log_softmax[idx] * -1 / batch_size));
    };
};

__global__ void diffNLLLoss(int batch_size, int *labels, float *dy)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(labels[idx]==1) dy[idx] = (float)(-1.f / batch_size);
    else dy[idx] = 0.f;
}



//////////////////////////////////
//* Accuracy from (log)softmax *//
//////////////////////////////////
__global__ void calAccuracy(int batch_size, int num_classes, float *softmax, int *labels, float *accuracy)
{
    extern __shared__ int prediction[];
    
    float max_value = -1000000;
    int num_trues = 0;
    int idx = threadIdx.x;   //batch_size
    for(int i=idx*num_classes; i<(idx+1)*num_classes; i++){   
        if(max_value < softmax[i]){
            max_value = softmax[i];
            prediction[idx] = (int)(i % num_classes);
        } 
    }
    __syncthreads();

    // reduction to max
    if(idx == 0){
        for(int i=0; i<batch_size; i++){
            if(prediction[i]==labels[i]) num_trues++;
        }
        *accuracy = (float)num_trues/batch_size;
    }
}
