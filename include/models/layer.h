#pragma once

#include <cudnn.h>
#include <cuda.h>
#include <string>
#include <cublas_v2.h>
#include <memory.h>

enum CudaLibrary
{
    CUDNN,
    CUBLAS
};

enum Processor
{
    GPU,
    CPU
};

struct ImageDto
{
    ImageDto(){};
    ImageDto(int n, int out_dims): batch_size(n), num_features(out_dims){};
    ImageDto(int n, int c, int h, int w): batch_size(n), num_channels(c), height(h), weight(w){
        num_features = c*h*w;
    }

    int batch_size = 0;
    int num_features = 0;
    int num_channels = 0;
    int height = 0; 
    int weight = 0;

    float *buffer_h = nullptr;
    float *buffer_d = nullptr;
};


class Layer
{
    protected:
        cudnnHandle_t cudnnHandle;
        cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
        cudnnFilterDescriptor_t filterDesc;
        cudnnConvolutionDescriptor_t convDesc;
        cudnnPoolingDescriptor_t poolingDesc;
        cudnnActivationDescriptor_t activationDesc;

        cublasHandle_t cublasHandle;
        
        int weight_len = 0;
        int bias_len = 0;
        void initWeightBias(unsigned int seed, std::string layer_name, int in_channels, int out_channels, int kernel_size);
        void initWeightBias(unsigned int seed, std::string layer_name, int in_features, int out_features);
        
    public:
        // Host input, weight & bias memory
        float *_input_h = nullptr;             // x
        float *_output_h = nullptr;            // y
        
        float *_weight_h = nullptr;            // w
        float *_bias_h = nullptr;              // b
        float *_grad_weight_h = nullptr;       // dw
        float *_grad_bias_h = nullptr;         // db
        
        // Device input, weight & bias memory
        float *_input_d = nullptr;              // x
        float *_output_d = nullptr;             // y

        float *_weight_d = nullptr;            // w
        float *_bias_d = nullptr;              // b
        float *_grad_weight_d = nullptr;       // dw
        float *_grad_bias_d = nullptr;         // db

        bool info_flag = 1;
        bool algo_flag = 1;

        void setCudnnDescriptor(cudnnHandle_t cudnnHandle,
                                cudnnTensorDescriptor_t srcTensorDesc, 
                                cudnnTensorDescriptor_t dstTensorDesc, 
                                cudnnTensorDescriptor_t biasTensorDesc,
                                cudnnFilterDescriptor_t filterDesc,
                                cudnnConvolutionDescriptor_t convDesc,
                                cudnnPoolingDescriptor_t poolingDesc,
                                cudnnActivationDescriptor_t activationDesc);
        void setCublasHandler(cublasHandle_t cublasHandle);

        void updateWeightBias(float learning_rate);
        
        virtual ImageDto Forward(ImageDto &data) = 0;
        virtual ImageDto Backward(ImageDto &data, int *labels) = 0;
};

class Conv2D: public Layer
{
    private:        
        cudnnConvolutionFwdAlgo_t conv_fwd_algo;
        cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo;
        cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo;

        cudnnConvolutionFwdAlgoPerf_t *fwd_algo_perf_results;
        cudnnConvolutionBwdFilterAlgoPerf_t *bwd_filter_algo_perf_results;
        cudnnConvolutionBwdDataAlgoPerf_t *bwd_data_algo_perf_results;


        
        size_t _fwd_workspace_bytes = 0;
        size_t _bwd_filter_workspace_bytes = 0;
        size_t _bwd_data_workspace_bytes = 0;
        void *_fwd_workspace_d = nullptr;
        void *_bwd_filter_workspace_d = nullptr;
        void *_bwd_data_workspace_d = nullptr;
        
        std::string _layer_name;
        int _in_channels;
        int _out_channels;
        int _kernel_size;
        int _stride;
        int _padding;
        int _dilation;    

        int in_n, in_c, in_h, in_w;
        int out_n, out_c, out_h, out_w;
        int num_features;

        void setConvAlgorithm();
        
    public:

        Conv2D(std::string layer_name, int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, int dilation=1);
        ~Conv2D();
        
        virtual ImageDto Forward(ImageDto &data);
        virtual ImageDto Backward(ImageDto &data, int *labels);
};


class Activation: public Layer
{
    private:
        cudnnActivationMode_t _activation_mode;

        std::string _layer_name;
        float _coef;

        int in_n, in_c, in_h, in_w;
        int out_n, out_c, out_h, out_w;
        int num_features;

        std::string prev_layer;
        
    public:
        Activation(std::string layer_name, cudnnActivationMode_t mode, float coef=0);
        ~Activation();

        virtual ImageDto Forward(ImageDto &data);
        virtual ImageDto Backward(ImageDto &data, int *labels);
};



class Pooling: public Layer
{
    private:
        cudnnPoolingMode_t _pooling_mode;

        std::string _layer_name;
        int _kernel_size;
        int _padding;
        int _stride;

        int in_n, in_c, in_h, in_w;
        int out_n, out_c, out_h, out_w;

    public:
        Pooling(std::string layer_name, int kernel_size, int stride, int padding, cudnnPoolingMode_t mode);
        ~Pooling();
    
        virtual ImageDto Forward(ImageDto &data);  
        virtual ImageDto Backward(ImageDto &data, int *labels);  
};


class Dense: public Layer
{
    private:
        std::string _layer_name;
        int _in_features;
        int _out_features;
        float *one_vec_d = nullptr;

    public:
        Dense(std::string layer_name, int in_features, int out_features);
        ~Dense();
        virtual ImageDto Forward(ImageDto &data);
        virtual ImageDto Backward(ImageDto &data, int *labels);
        
};

class Softmax: public Layer
{
    private:
        std::string _layer_name;
        cudnnSoftmaxAlgorithm_t _softmax_algo = CUDNN_SOFTMAX_ACCURATE;
        cudnnSoftmaxMode_t _softmax_mode = CUDNN_SOFTMAX_MODE_CHANNEL;

        CudaLibrary backward_library = CUBLAS;
        Processor dy_cal = GPU;

    public:
        Softmax(std::string layer_name);
        Softmax(std::string layer_name, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode);
        
        virtual ImageDto Forward(ImageDto &data);
        virtual ImageDto Backward(ImageDto &data, int *labels);
};

