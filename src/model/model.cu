#include <cudnn.h>
#include <cuda.h>
#include <iostream>

#include "model.h"
#include "utils.h"
#include "loss.h"


void Model::createHandles()
{
    checkCUDNN(cudnnCreate(&cudnnHandle));        
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
    checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));

    checkCUBLAS(cublasCreate(&cublasHandle));
}   

void Model::destroyHandles()
{
    checkCUDNN(cudnnDestroy(cudnnHandle));        
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
    checkCUDNN(cudnnDestroyActivationDescriptor(activationDesc));    
}

Model::Model(){
    createHandles();
}

Model::~Model(){
    destroyHandles();
}

void Model::addLayers(Layer *layer)
{
    (*layer).setCudnnDescriptor(cudnnHandle, 
                               srcTensorDesc, 
                               dstTensorDesc, 
                               biasTensorDesc,
                               filterDesc, 
                               convDesc, 
                               poolingDesc, 
                               activationDesc);
                               
    (*layer).setCublasHandler(cublasHandle);
    model.push_back(layer);
}

ImageDto Model::Forward(ImageDto &data)
{
    ImageDto output = data;
    for(auto layer : model)
    {
        output = (*layer).Forward(output);
        checkCUDA(cudaDeviceSynchronize());
    }
    return output;
}

ImageDto Model::Backward(ImageDto &data, int *labels)
{
    ImageDto output = data;
    for(auto layer = model.crbegin(); layer != model.crend(); layer++)
    {
        output = (*layer)->Backward(output, labels);//output = (*layer).Backward(output, labels);
        (*layer)->info_flag = 0;
    }
    return output;
}

void Model::Update(float learning_rate)
{
    for(auto layer : model)
    {
        (*layer).updateWeightBias(learning_rate);      
    }
}

float Model::Loss(ImageDto &data, int *onehot_labels_d, cudnnSoftmaxAlgorithm_t softmax_algo)
{
    if(loss == nullptr) checkCUDA(cudaMallocManaged(&loss, sizeof(float)));
    *loss = 0.f;
    
    if(softmax_algo==CUDNN_SOFTMAX_LOG){
        nLLLoss<<<data.batch_size, data.num_features>>>(data.batch_size, data.buffer_d, onehot_labels_d, loss);
    }else{
        nLLoss<<<data.batch_size, data.num_features>>>(data.batch_size, data.buffer_d, onehot_labels_d, loss);
    }
    checkCUDA(cudaDeviceSynchronize());
    
    return *loss;
}

float Model::Accuracy(ImageDto &data, int *labels_d, int num_classes)
{
    if(accuracy == nullptr) checkCUDA(cudaMallocManaged(&accuracy, sizeof(float)));
    *accuracy = 0.f;

    calAccuracy<<<1, data.batch_size, data.batch_size>>>(data.batch_size, num_classes, data.buffer_d, labels_d, accuracy); // batchsize should be better for multiple of 32, and less than 2048.
    checkCUDA(cudaDeviceSynchronize());

    return *accuracy;
}