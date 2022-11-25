#pragma once
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>

#include "layer.h"

class Model
{
    public:
        Model();       
        ~Model();

        void createHandles();
        void destroyHandles();
        void addLayers(Layer *layer);
        void Update(float learning_rate);

        ImageDto Forward(ImageDto &data); 
        ImageDto Backward(ImageDto &data, int *labels);
        
        float Loss(ImageDto &data, int *onehot_labels, cudnnSoftmaxAlgorithm_t softmax_algo);
        float Accuracy(ImageDto &data, int *labels, int num_classes);
 
        std::vector<Layer*> model;

    protected:
        cudnnHandle_t cudnnHandle;
        cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
        cudnnFilterDescriptor_t filterDesc;
        cudnnConvolutionDescriptor_t convDesc;
        cudnnPoolingDescriptor_t poolingDesc;
        cudnnActivationDescriptor_t activationDesc;
        cublasHandle_t cublasHandle;

    private:
        float *loss = nullptr;    
        float *accuracy = nullptr;
};