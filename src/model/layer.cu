#include <cudnn.h>
#include <cuda.h>
#include <cublas.h>
#include <random>
#include <iostream>
#include <string>
#include <math.h>
// #include <cudnn_frontend.h>
#include "utils.h"
#include "layer.h"
#include "loss.h"

/* For conv2D layer */
void Layer::initWeightBias(unsigned int seed, std::string layer_name, int in_channels, int out_channels, int kernel_size)
{
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    weight_len = in_channels * out_channels * kernel_size * kernel_size;
    bias_len = out_channels;
    _weight_h = (float*)malloc(sizeof(float) * weight_len);
    _bias_h = (float*)malloc(sizeof(float) * bias_len);

    // // He initialization
    int n_in = in_channels * kernel_size * kernel_size;
    float range = sqrt(6.f / n_in);
    std::uniform_real_distribution<> dis(-range, range);
    for(int i=0; i<weight_len; i++){
        _weight_h[i] = static_cast<float>(dis(gen));
    }
    for(int i=0; i<bias_len; i++){
        _bias_h[i] = 0.f;
    }

    // memcpy from host to cuda
    checkCUDA(cudaMalloc(&_weight_d, sizeof(float) * weight_len));
    checkCUDA(cudaMalloc(&_bias_d, sizeof(float) * bias_len));
    checkCUDA(cudaMemcpy(_weight_d, _weight_h, sizeof(float) * weight_len, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(_bias_d, _bias_h, sizeof(float) * bias_len, cudaMemcpyHostToDevice));

    std::cout << "   >> [" << layer_name << "]" << " weights and biases are initialized." << std::endl;
}

/* For dense layer */
void Layer::initWeightBias(unsigned int seed, std::string layer_name, int in_features, int out_features)
{
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    weight_len = in_features * out_features;
    bias_len = out_features;
    _weight_h = (float*)malloc(sizeof(float) * weight_len);
    _bias_h = (float*)malloc(sizeof(float) * bias_len);

    // He initialization
    float range = sqrt(6.f / in_features);
    std::uniform_real_distribution<> dis(-range, range);
    for(int i=0; i<weight_len; i++){
        _weight_h[i] = static_cast<float>(dis(gen));
    }
    for(int i=0; i< bias_len; i++){
        _bias_h[i] = 0.f;
    }

    checkCUDA(cudaMalloc(&_weight_d, sizeof(float) * weight_len));
    checkCUDA(cudaMalloc(&_bias_d, sizeof(float) * bias_len));
    checkCUDA(cudaMemcpy(_weight_d, _weight_h, sizeof(float) * weight_len, cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(_bias_d, _bias_h, sizeof(float) * bias_len, cudaMemcpyHostToDevice));

    std::cout << "   >> [" << layer_name << "]" << " weights and biases are initialized." << std::endl;
}

void Layer::updateWeightBias(float learning_rate)
{
    float lr = -1 * learning_rate; 
    /* Stochastic gradient descent */
    //w = w - lr * dW
    checkCUBLAS(cublasSaxpy(cublasHandle, weight_len,
                            &lr,
                            _grad_weight_d, 1,
                            _weight_d, 1));

    //b = b - lr * db
    checkCUBLAS(cublasSaxpy(cublasHandle, bias_len,
                            &lr,
                            _grad_bias_d, 1,
                            _bias_d, 1));
}

void Layer::setCudnnDescriptor(cudnnHandle_t handler, 
                               cudnnTensorDescriptor_t src, 
                               cudnnTensorDescriptor_t dst,
                               cudnnTensorDescriptor_t bias,
                               cudnnFilterDescriptor_t filter,
                               cudnnConvolutionDescriptor_t conv,
                               cudnnPoolingDescriptor_t pooling,
                               cudnnActivationDescriptor_t activation)
{
    cudnnHandle = handler;
    srcTensorDesc = src;
    dstTensorDesc = dst;
    biasTensorDesc = bias;
    filterDesc = filter;
    convDesc = conv;
    poolingDesc = pooling;
    activationDesc = activation;
}

void Layer::setCublasHandler(cublasHandle_t handler)
{
    cublasHandle = handler;
}

/////////////////////////
//* Convolution layer *//
/////////////////////////
Conv2D::Conv2D(std::string layer_name, int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation):_layer_name(layer_name), _in_channels(in_channels), _out_channels(out_channels), _kernel_size(kernel_size), _stride(stride), _padding(padding), _dilation(dilation)
{ 
    initWeightBias(1, _layer_name, _in_channels, _out_channels, _kernel_size); // Initialize weight and bias
}
Conv2D::~Conv2D(){}

void Conv2D::setConvAlgorithm()
{
    std::cout << ">> Searching for the fastest convolution algorithm ... " << std::endl;
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount = -1;
    int maxAlgoCount = 0;

    /////////////////////////
    //* Forward algorithm *//
    /////////////////////////
    std::cout << "   >> FWD algorithm" << std::endl;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxAlgoCount));
    fwd_algo_perf_results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t) * maxAlgoCount);

    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle, 
                                                      srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 
                                                      requestedAlgoCount, &returnedAlgoCount, fwd_algo_perf_results));
    for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
        printf("   ===== %s for Algo %d: %f time requiring %llu memory\n", 
        cudnnGetErrorString(fwd_algo_perf_results[algoIndex].status), 
                            fwd_algo_perf_results[algoIndex].algo, 
                            fwd_algo_perf_results[algoIndex].time, 
                            (unsigned long long)fwd_algo_perf_results[algoIndex].memory);
    }
    conv_fwd_algo = fwd_algo_perf_results[0].algo;
    std::cout << "   >> Fastest FWD algorithm:: Algo " << conv_fwd_algo << std::endl;

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, 
                                                       srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 
                                                       conv_fwd_algo, &_fwd_workspace_bytes));
    checkCUDA(cudaMalloc(&_fwd_workspace_d, _fwd_workspace_bytes));
    std::cout << "   >> Memory buffer size required:: " << _fwd_workspace_bytes << std::endl;
    std::cout << "" << std::endl;

    //////////////////////////
    //* Backward algorithm *//
    //////////////////////////
    // filter
    //Errors if it is defined in a dfferent function, don't know why.
    std::cout << "   >> BWD Filter algorithm" << std::endl;
    requestedAlgoCount = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle, &maxAlgoCount));
    bwd_filter_algo_perf_results = (cudnnConvolutionBwdFilterAlgoPerf_t*)malloc(sizeof(cudnnConvolutionBwdFilterAlgoPerf_t) * maxAlgoCount);
    
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle, 
                                                             srcTensorDesc, dstTensorDesc, convDesc, filterDesc, 
                                                             requestedAlgoCount, &returnedAlgoCount, bwd_filter_algo_perf_results));                                                                
    for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
        printf("   ===== %s for Algo %d: %f time requiring %llu memory\n", 
        cudnnGetErrorString(bwd_filter_algo_perf_results[algoIndex].status), 
                            bwd_filter_algo_perf_results[algoIndex].algo, 
                            bwd_filter_algo_perf_results[algoIndex].time, 
                            (unsigned long long)bwd_filter_algo_perf_results[algoIndex].memory);
    }
    conv_bwd_filter_algo = bwd_filter_algo_perf_results[0].algo;
    std::cout << "   >> Fastest BWD Filter algorithm:: Algo " << conv_bwd_filter_algo << std::endl;    

    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, 
                                                              srcTensorDesc, dstTensorDesc, convDesc, filterDesc, 
                                                              conv_bwd_filter_algo, &_bwd_filter_workspace_bytes));
    checkCUDA(cudaMalloc(&_bwd_filter_workspace_d, _bwd_filter_workspace_bytes));
    std::cout << "   >> Memory Buffer size required:: " << _bwd_filter_workspace_bytes << std::endl;
    std::cout << "" << std::endl;

    // data
    std::cout << "   >> BWD Data algorithm" << std::endl;
    requestedAlgoCount = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle, &maxAlgoCount));
    bwd_data_algo_perf_results = (cudnnConvolutionBwdDataAlgoPerf_t*)malloc(sizeof(cudnnConvolutionBwdDataAlgoPerf_t) * maxAlgoCount);

    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle, 
                                                           filterDesc, dstTensorDesc, convDesc, srcTensorDesc, 
                                                           requestedAlgoCount, &returnedAlgoCount, bwd_data_algo_perf_results));
    for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
        printf("   ===== %s for Algo %d: %f time requiring %llu memory\n", 
        cudnnGetErrorString(bwd_data_algo_perf_results[algoIndex].status), 
                            bwd_data_algo_perf_results[algoIndex].algo, 
                            bwd_data_algo_perf_results[algoIndex].time, 
                            (unsigned long long)bwd_data_algo_perf_results[algoIndex].memory);
    }
    conv_bwd_data_algo = bwd_data_algo_perf_results[0].algo;
    std::cout << "   >> Fastest Bwd Data Algorithm:: Algo " << conv_bwd_data_algo << std::endl;  

    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, 
                                                            filterDesc, dstTensorDesc, convDesc, srcTensorDesc, 
                                                            conv_bwd_data_algo, &_bwd_data_workspace_bytes));  
    checkCUDA(cudaMalloc(&_bwd_data_workspace_d, _bwd_data_workspace_bytes));
    std::cout << "   >> Memory Buffer size required:: " << _bwd_data_workspace_bytes << std::endl;
    std::cout << "" << std::endl;
}

ImageDto Conv2D::Forward(ImageDto &input){
    num_features = input.num_features;
    in_n = input.batch_size;
    in_c = input.num_channels;
    in_h = input.height;
    in_w = input.weight;
    
    // Persist input
    if(_input_h == nullptr) _input_h = (float*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
    if(_input_d == nullptr) checkCUDA(cudaMalloc(&_input_d, sizeof(float) * in_n * in_c * in_h * in_w));
    memcpy(_input_h, input.buffer_h, sizeof(float) * in_n * in_c * in_h * in_w);
    checkCUDA(cudaMemcpy(_input_d, _input_h, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyHostToDevice));

    // Free input DTO memory
    free(input.buffer_h);
    checkCUDA(cudaFree(input.buffer_d));
    
    if(info_flag) std::cout << "FWD:: [Layer Name:: " << _layer_name << "]" << std::endl;   
    //* Initialize *//    
    // Set input shape
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
    if(info_flag) printf(">> Convolution input shape:: %d x %d x %d x %d\n", in_n, in_c, in_h, in_w);
    // Set kernel filter shape
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, _out_channels, _in_channels, _kernel_size, _kernel_size));
    // Set conv layer structure
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, _padding, _padding, _stride, _stride, _dilation, _dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // Get output shape from convolution operation
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
    if(info_flag) printf(">> Convolution output shape:: %d x %d x %d x %d\n\n", out_n, out_c, out_h, out_w);
    // Set output shape
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    // Set bias shape
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_c, 1, 1));
    
    //* Set conv computing algorithm *//
    if(algo_flag){
        setConvAlgorithm();
        algo_flag = 0;
    }

    // Set output DTO
    ImageDto output = ImageDto(out_n, out_c, out_h, out_w);
    output.buffer_h = (float*)malloc(sizeof(float) * (out_n * out_c * out_h * out_w));
    checkCUDA(cudaMalloc(&(output.buffer_d), sizeof(float) * (out_n * out_c * out_h * out_w)));   
    
    // cudnnConvolutionBiasActivationForward selects slower kernel so that seprate implementation would be better.
    float conv_alpha = 1;
    float conv_beta = 0;
    checkCUDNN(cudnnConvolutionForward(cudnnHandle, 
                                       &conv_alpha, 
                                       srcTensorDesc, _input_d, 
                                       filterDesc, _weight_d, 
                                       convDesc, conv_fwd_algo, _fwd_workspace_d, _fwd_workspace_bytes, 
                                       &conv_beta, 
                                       dstTensorDesc, output.buffer_d));

    // bias doesn't need to be [out_c * out_h * out_w]
    float bias_alpha = 1;
    float bias_beta = 1;
    checkCUDNN(cudnnAddTensor(cudnnHandle,
                              &bias_alpha, 
                              biasTensorDesc, _bias_d,
                              &bias_beta,
                              dstTensorDesc, output.buffer_d));

    // Persist outputs
    if(_output_h == nullptr) _output_h = new float[out_n * out_c * out_h * out_w];    
    checkCUDA(cudaMemcpy(_output_h, output.buffer_d, sizeof(float) * (out_n * out_c * out_h * out_w), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(output.buffer_h, output.buffer_d, sizeof(float) * (out_n * out_c * out_h * out_w), cudaMemcpyDeviceToHost));

    return output;
}

ImageDto Conv2D::Backward(ImageDto &dy, int *labels)
{
    // Set dW
    if(_grad_weight_h == nullptr) _grad_weight_h = (float*)malloc(sizeof(float) * (_out_channels * _kernel_size * _kernel_size * _in_channels));
    if(_grad_weight_d == nullptr) checkCUDA(cudaMalloc(&_grad_weight_d, sizeof(float) * (_out_channels * _kernel_size * _kernel_size * _in_channels)));

    // Set db
    if(_grad_bias_h == nullptr) _grad_bias_h = (float*)malloc(sizeof(float) *_out_channels);
    if(_grad_bias_d == nullptr) checkCUDA(cudaMalloc(&_grad_bias_d, sizeof(float) * _out_channels));
    
    // Set dx
    ImageDto dx = ImageDto(in_n, in_c, in_h, in_w);
    dx.buffer_h = (float*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
    checkCUDA(cudaMalloc(&(dx.buffer_d), sizeof(float) * in_n * in_c * in_h * in_w));
        
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_c, 1, 1));       
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, _out_channels, _in_channels, _kernel_size, _kernel_size));

    // Conv bias backward
    float bias_alpha = 1;
    float bias_beta = 0;
    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, 
                                            &bias_alpha, 
                                            dstTensorDesc, dy.buffer_d,         //dy
                                            &bias_beta, 
                                            biasTensorDesc, _grad_bias_d));     //db
    checkCUDA(cudaMemcpy(_grad_bias_h, _grad_bias_d, sizeof(float) * _out_channels, cudaMemcpyDeviceToHost));
    
    // Conv filter backward
    float filter_alpha = 1;
    float filter_beta = 0;
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,
                                              &filter_alpha,
                                              srcTensorDesc, _input_d,          //x
                                              dstTensorDesc, dy.buffer_d,       //dy
                                              convDesc, conv_bwd_filter_algo, _bwd_filter_workspace_d, _bwd_filter_workspace_bytes,
                                              &filter_beta,
                                              filterDesc, _grad_weight_d));     //dW
    checkCUDA(cudaMemcpy(_grad_weight_h, _grad_weight_d, sizeof(float) * _out_channels * _in_channels * _kernel_size * _kernel_size, cudaMemcpyDeviceToHost));

    // Conv backward
    float conv_alpha = 1;
    float conv_beta = 0;
    checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
                                            &conv_alpha,
                                            filterDesc, _weight_d,              //w
                                            dstTensorDesc, dy.buffer_d,         //Dy
                                            convDesc, conv_bwd_data_algo, _bwd_data_workspace_d, _bwd_data_workspace_bytes, 
                                            &conv_beta,
                                            srcTensorDesc, dx.buffer_d));       //dx
    checkCUDA(cudaMemcpy(dx.buffer_h, dx.buffer_d, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyDeviceToHost));
    
    // Free input DTO memory
    free(dy.buffer_h);
    checkCUDA(cudaFree(dy.buffer_d));
    return dx;
}


////////////////////////
//* Activation layer *//
////////////////////////
Activation::Activation(std::string layer_name, cudnnActivationMode_t mode, float coef): _layer_name(layer_name), _activation_mode(mode), _coef(coef){}
Activation::~Activation(){}
ImageDto Activation::Forward(ImageDto &input)
{
    num_features = input.num_features;
    in_n = input.batch_size;
    in_c = input.num_channels;
    in_h = input.height;
    in_w = input.weight;
    out_n = in_n;
    out_c = in_c; 
    out_h = in_h; 
    out_w = in_w;
    
    // Persist input
    if(_input_h == nullptr) _input_h = (float*)malloc(sizeof(float) * in_n * num_features);
    if(_input_d == nullptr) checkCUDA(cudaMalloc(&_input_d, sizeof(float) * in_n * num_features));
    memcpy(_input_h, input.buffer_h, sizeof(float) * in_n * num_features);
    checkCUDA(cudaMemcpy(_input_d, _input_h, sizeof(float) * in_n * num_features, cudaMemcpyHostToDevice));

    if(info_flag) std::cout << "FWD:: [Layer Name:: " << _layer_name << "]" << std::endl;
    ImageDto output;
    //* Initialize *//
    if(input.num_channels == 0){
        prev_layer="dense";
        // Set input shape from Dense
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, num_features, 1, 1));
        if(info_flag) printf(">> Activation input shape:: %d x %d\n", in_n, num_features);
        // Set Activation function
        checkCUDNN(cudnnSetActivationDescriptor(activationDesc, _activation_mode, CUDNN_PROPAGATE_NAN, _coef));
        // Set output shape
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, num_features, 1, 1));
        if(info_flag) printf(">> Activation output shape:: %d x %d\n\n", out_n, num_features);

        //set output DTO    
        output = ImageDto(out_n, num_features);
        output.buffer_h = (float*)malloc(sizeof(float) * (out_n * num_features));
        checkCUDA(cudaMalloc(&(output.buffer_d), sizeof(float) * (out_n * num_features)));    
    }else{
        prev_layer="conv";
        // Set input shape from CNN
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
        if(info_flag) printf(">> Activation input shape:: %d x %d x %d x %d\n", in_n, in_c, in_h, in_w);
        // Set Activation function
        checkCUDNN(cudnnSetActivationDescriptor(activationDesc, _activation_mode, CUDNN_PROPAGATE_NAN, _coef));
        // Set output shape
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
        if(info_flag) printf(">> Activation output shape:: %d x %d x %d x %d\n\n", out_n, out_c, out_h, out_w);
            
        // set output DTO
        output = ImageDto(out_n, out_c, out_h, out_w);
        output.buffer_h = (float*)malloc(sizeof(float) * (out_n * out_c * out_h * out_w));
        checkCUDA(cudaMalloc(&(output.buffer_d), sizeof(float) * (out_n * out_c * out_h * out_w)));    
    }
    
    // Free input DTO memory
    free(input.buffer_h);
    checkCUDA(cudaFree(input.buffer_d));

    float activation_alpha = 1;
    float activation_beta = 0;
    checkCUDNN(cudnnActivationForward(cudnnHandle, 
                                      activationDesc, 
                                      &activation_alpha, 
                                      srcTensorDesc, _input_d, 
                                      &activation_beta, 
                                      dstTensorDesc, output.buffer_d));
    // Persist outputs
    if(_output_h == nullptr) _output_h = new float[out_n *num_features];
    checkCUDA(cudaMemcpy(_output_h, output.buffer_d, sizeof(float) * (out_n * num_features), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(output.buffer_h, output.buffer_d, sizeof(float) * (out_n * num_features), cudaMemcpyDeviceToHost));

    return output;
}

ImageDto Activation::Backward(ImageDto &dy, int *labels)
{
    // Set y
    if(_output_d == nullptr) cudaMalloc(&_output_d, sizeof(float) * out_n * num_features);
    cudaMemcpy(_output_d, _output_h, sizeof(float) * out_n * num_features, cudaMemcpyHostToDevice);

    // Set dx
    ImageDto dx;
    if(prev_layer=="dense"){
        dx = ImageDto(in_n, num_features);
        dx.buffer_h = (float*)malloc(sizeof(float) * in_n * num_features);
        checkCUDA(cudaMalloc(&(dx.buffer_d), sizeof(float) * in_n * num_features));

        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, num_features, 1, 1));
        checkCUDNN(cudnnSetActivationDescriptor(activationDesc, _activation_mode, CUDNN_PROPAGATE_NAN, _coef));
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, num_features, 1, 1));
    }else{
        dx = ImageDto(in_n, in_c, in_h, in_w);
        dx.buffer_h = (float*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
        checkCUDA(cudaMalloc(&(dx.buffer_d), sizeof(float) * in_n * in_c * in_h * in_w));
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
        checkCUDNN(cudnnSetActivationDescriptor(activationDesc, _activation_mode, CUDNN_PROPAGATE_NAN, _coef));
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    }
    // Activation backward
    float activation_alpha = 1;
    float activation_beta = 0;
    checkCUDNN(cudnnActivationBackward(cudnnHandle, 
                                       activationDesc,
                                       &activation_alpha,
                                       dstTensorDesc, _output_d,                //y
                                       dstTensorDesc, dy.buffer_d,              //dy
                                       srcTensorDesc, _input_d,                 //x
                                       &activation_beta,
                                       srcTensorDesc, dx.buffer_d));            //dx
    checkCUDA(cudaMemcpy(dx.buffer_h, dx.buffer_d, sizeof(float) * in_n * num_features, cudaMemcpyDeviceToHost));
    
    // Free input DTO memory
    free(dy.buffer_h);
    checkCUDA(cudaFree(dy.buffer_d));
    
    return dx;
}


////////////////////////
//* Pooling layer *//
////////////////////////
Pooling::Pooling(std::string layer_name, int kernel_size, int stride, int padding, cudnnPoolingMode_t mode):_layer_name(layer_name), _kernel_size(kernel_size), _padding(padding), _stride(stride), _pooling_mode(mode){}

Pooling::~Pooling(){}
ImageDto Pooling::Forward(ImageDto &input){
    in_n = input.batch_size;
    in_c = input.num_channels;
    in_h = input.height;
    in_w = input.weight;

    // Persist input
    if(_input_h == nullptr) _input_h = (float*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
    if(_input_d == nullptr) checkCUDA(cudaMalloc(&_input_d, sizeof(float) * in_n * in_c * in_h * in_w));
    memcpy(_input_h, input.buffer_h, sizeof(float) * in_n * in_c * in_h * in_w);
    checkCUDA(cudaMemcpy(_input_d, _input_h, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyHostToDevice));

    // Free input DTO
    free(input.buffer_h);
    checkCUDA(cudaFree(input.buffer_d));

    if(info_flag) std::cout << "FWD:: [Layer Name:: " << _layer_name << "]" << std::endl;
    // Initialize 
    // Set input shape
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
    if(info_flag) printf(">> Pooling input shape:: %d x %d x %d x %d\n", in_n, in_c, in_h, in_w);
    // Set pooling layer structure
    checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, _pooling_mode, CUDNN_PROPAGATE_NAN, _kernel_size, _kernel_size, _padding, _padding, _stride, _stride));
    // Get output shape from pooling operation
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolingDesc, srcTensorDesc, &out_n, &out_c, &out_h, &out_w));
    if(info_flag) printf(">> Pooling output shape:: %d x %d x %d x %d\n\n", out_n, out_c, out_h, out_w);
    // Set output shape
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    
    // Set output DTO
    ImageDto output = ImageDto(out_n, out_c, out_h, out_w);
    output.buffer_h = (float*)malloc(sizeof(float) * (out_n * out_c * out_h * out_w));
    checkCUDA(cudaMalloc(&(output.buffer_d), sizeof(float) * (out_n * out_c * out_h * out_w)));    

    // Pooling forward
    float alpha = 1;
    float beta = 0;
    checkCUDNN(cudnnPoolingForward(cudnnHandle, 
                                   poolingDesc, 
                                   &alpha, 
                                   srcTensorDesc, _input_d,                 //x
                                   &beta, 
                                   dstTensorDesc, output.buffer_d));        //y

    // Persist outputs
    if(_output_h == nullptr) _output_h = new float[out_n * out_c * out_h * out_w];
    checkCUDA(cudaMemcpy(_output_h, output.buffer_d, sizeof(float) * (out_n * out_c * out_h * out_w), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(output.buffer_h, output.buffer_d, sizeof(float) * (out_n * out_c * out_h * out_w), cudaMemcpyDeviceToHost));

    return output;
}

ImageDto Pooling::Backward(ImageDto &dy, int *labels)
{ 
    // Set y
    if(_output_d == nullptr) checkCUDA(cudaMalloc(&_output_d, sizeof(float) * out_n * out_c * out_h * out_w));
    checkCUDA(cudaMemcpy(_output_d, _output_h, sizeof(float) * out_n * out_c * out_h * out_w, cudaMemcpyHostToDevice));
   
    // Set dx
    ImageDto dx = ImageDto(in_n, in_c, in_h, in_w);
    dx.buffer_h = (float*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
    checkCUDA(cudaMalloc(&(dx.buffer_d), sizeof(float) * in_n * in_c * in_h * in_w));

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc, _pooling_mode, CUDNN_PROPAGATE_NAN, _kernel_size, _kernel_size, _padding, _padding, _stride, _stride));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    
    // Pooling backward
    float pooling_alpha = 1;
    float pooling_beta = 0;
    checkCUDNN(cudnnPoolingBackward(cudnnHandle,
                                    poolingDesc,
                                    &pooling_alpha,
                                    dstTensorDesc, _output_d,           //y
                                    dstTensorDesc, dy.buffer_d,         //dy
                                    srcTensorDesc, _input_d,            //x
                                    &pooling_beta,
                                    srcTensorDesc, dx.buffer_d));       //dx
    checkCUDA(cudaMemcpy(dx.buffer_h, dx.buffer_d, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyDeviceToHost));

    // Free input DTO memory
    free(dy.buffer_h);
    checkCUDA(cudaFree(dy.buffer_d));
    
    return dx;
}



/////////////////////
//* Dense layer *////
/////////////////////
Dense::Dense(std::string layer_name, int in_features, int out_features):_layer_name(layer_name), _in_features(in_features), _out_features(out_features)
{
    initWeightBias(1, _layer_name, _in_features, _out_features); // Initialize weight and bias 
}

ImageDto Dense::Forward(ImageDto &input)
{
    int batch_size = input.batch_size;

    // Persist input
    if(_input_h == nullptr) _input_h = (float*)malloc(sizeof(float) * batch_size * _in_features);
    if(_input_d == nullptr) checkCUDA(cudaMalloc(&_input_d, sizeof(float) * batch_size * _in_features));
    memcpy(_input_h, input.buffer_h, sizeof(float) * batch_size * _in_features);
    checkCUDA(cudaMemcpy(_input_d, _input_h, sizeof(float) * batch_size * _in_features, cudaMemcpyHostToDevice));

    // Free input DTO memory
    free(input.buffer_h);
    checkCUDA(cudaFree(input.buffer_d));

    if(info_flag) std::cout << "FWD:: [Layer Name:: " << _layer_name << "]" << std::endl;
    if(info_flag) printf(">> Dense input shape:: %d x %d\n", batch_size, _in_features);
    if(info_flag) printf(">> Dense output shape:: %d x %d\n\n", batch_size, _out_features);

    // Set output DTO
    ImageDto output = ImageDto(batch_size, _out_features);
    output.buffer_h = (float*)malloc(sizeof(float) * batch_size * _out_features);
    checkCUDA(cudaMalloc(&(output.buffer_d), sizeof(float) * batch_size * _out_features));
    
    /* x shape: [batch_size, in_fetures], Weight shape: [out_feature, in_features]     
    y = x * W(T)   ==>  y(T) = W * x(T) */
    float dense_alpha = 1;
    float dense_beta = 0;
    checkCUBLAS(cublasSgemm(cublasHandle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            _out_features, batch_size, _in_features,
                            &dense_alpha, 
                            _weight_d, _in_features,                    //W
                            _input_d, _in_features,                     //x
                            &dense_beta,
                            output.buffer_d, _out_features));           //y
    
    // Define one_vector for bias
    if(one_vec_d == nullptr){
        checkCUDA(cudaMalloc(&one_vec_d, sizeof(float) * batch_size));
        getOneVec<<<1, batch_size>>>(one_vec_d, batch_size);
        checkCUDA(cudaDeviceSynchronize());
    }

    // (batch_sizex1) x (1xbias) = batch x bias   -->   y(T) = (bias * batch_size) + y(T)
    float bias_alpha = 1;
    float bias_beta = 1;
    checkCUBLAS(cublasSgemm(cublasHandle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            _out_features, batch_size, 1,
                            &bias_alpha, 
                            _bias_d, _out_features,                     // bias
                            one_vec_d, 1,                               // 1 x bias
                            &bias_beta,
                            output.buffer_d, _out_features));           //y

    // Persist outputs
    if(_output_h == nullptr) _output_h = new float[batch_size * _out_features];
    checkCUDA(cudaMemcpy(_output_h, output.buffer_d, sizeof(float) * (batch_size * _out_features), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(output.buffer_h, output.buffer_d, sizeof(float) * (batch_size * _out_features), cudaMemcpyDeviceToHost));

    return output;
}

ImageDto Dense::Backward(ImageDto &dy, int *labels)
{
    int batch_size = dy.batch_size;
    
    // Set dW
    if(_grad_weight_h == nullptr) _grad_weight_h = (float*)malloc(sizeof(float) * (_out_features * _in_features));
    if(_grad_weight_d == nullptr) checkCUDA(cudaMalloc(&_grad_weight_d, sizeof(float) * _out_features * _in_features));

    // Set db
    if(_grad_bias_h == nullptr) _grad_bias_h = (float*)malloc(sizeof(float) *_out_features);
    if(_grad_bias_d == nullptr) checkCUDA(cudaMalloc(&_grad_bias_d, sizeof(float) * _out_features));

    // Set dx 
    ImageDto dx = ImageDto(batch_size, _in_features);
    dx.buffer_h = (float*)malloc(sizeof(float) * batch_size * _in_features);
    checkCUDA(cudaMalloc(&dx.buffer_d, sizeof(float) * batch_size * _in_features));

    // Define one_vector for bias
    if(one_vec_d == nullptr){
        checkCUDA(cudaMalloc(&one_vec_d, sizeof(float) * batch_size));
        getOneVec<<<1, batch_size>>>(one_vec_d, batch_size);
        checkCUDA(cudaDeviceSynchronize());
    }

    /* db = one_vec(T) * dy   --> dy(T) * one_vec */
    float bias_alpha = 1;
    float bias_beta = 0;
    checkCUBLAS(cublasSgemv(cublasHandle, 
                            CUBLAS_OP_N,
                            _out_features, batch_size,
                            &bias_alpha, 
                            dy.buffer_d, _out_features,         //dy
                            one_vec_d, 1,                       // b
                            &bias_beta,
                            _grad_bias_d, 1));                  //db
    checkCUDA(cudaMemcpy(_grad_bias_h, _grad_bias_d, sizeof(float) * _out_features, cudaMemcpyDeviceToHost));
            
    /* dW = dy(T) * x   -->  dW(T) = x(T) * dy */
    float dense_alpha = 1;
    float dense_beta = 0;
    checkCUBLAS(cublasSgemm(cublasHandle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            _in_features, _out_features, batch_size,
                            &dense_alpha,
                            _input_d, _in_features,             //x
                            dy.buffer_d, _out_features,         //dy
                            &dense_beta,
                            _grad_weight_d, _in_features));     //dw
    checkCUDA(cudaMemcpy(_grad_weight_h, _grad_weight_d, sizeof(float) * (_out_features * _in_features), cudaMemcpyDeviceToHost));

    /* dx = dy * W  -->  dx(T) = W(T) * dy(T) */
    checkCUBLAS(cublasSgemm(cublasHandle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            _in_features, batch_size, _out_features,
                            &dense_alpha,
                            _weight_d, _in_features,            //W
                            dy.buffer_d, _out_features,         //dy
                            &dense_beta,
                            dx.buffer_d, _in_features));        //dx
    checkCUDA(cudaMemcpy(dx.buffer_h, dx.buffer_d, sizeof(float) * batch_size * _in_features, cudaMemcpyDeviceToHost));

    // Free input DTO memory
    free(dy.buffer_h);
    checkCUDA(cudaFree(dy.buffer_d));

    return dx;
}


/////////////////////
//* Softmax layer *//
/////////////////////
Softmax::Softmax(std::string layer_name):_layer_name(layer_name){};
Softmax::Softmax(std::string layer_name, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode):_layer_name(layer_name), _softmax_algo(algo), _softmax_mode(mode){};
ImageDto Softmax::Forward(ImageDto &input)
{
    int batch_size = input.batch_size;
    int num_features = input.num_features;

    // Persist input
    if(_input_h == nullptr) _input_h = (float*)malloc(sizeof(float) * batch_size * num_features);
    if(_input_d == nullptr) checkCUDA(cudaMalloc(&_input_d, sizeof(float) * batch_size * num_features));
    memcpy(_input_h, input.buffer_h, sizeof(float) * batch_size * num_features);
    checkCUDA(cudaMemcpy(_input_d, _input_h, sizeof(float) * batch_size * num_features, cudaMemcpyHostToDevice));

    // Free input DTO memory
    free(input.buffer_h);
    checkCUDA(cudaFree(input.buffer_d));

    if(info_flag) std::cout << "FWD:: [Layer Name:: " << _layer_name << "]" << std::endl;
    // Set input shape
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, num_features, 1, 1));
    if(info_flag) printf(">> Softmax input shape:: %d x %d\n", batch_size, num_features);
    // Set output shape
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, num_features, 1, 1));
    if(info_flag) printf(">> Softmax output shape:: %d x %d\n\n", batch_size, num_features);

    // Set output DTO memory
    ImageDto output = ImageDto(batch_size, num_features);
    output.buffer_h = (float*)malloc(sizeof(float) * batch_size * num_features);
    cudaMalloc(&(output.buffer_d), sizeof(float) * batch_size * num_features);

    // Softmax forward
    float softmax_alpha = 1;
    float softmax_beta = 0;
    checkCUDNN(cudnnSoftmaxForward(cudnnHandle,
                                   _softmax_algo, 
                                   _softmax_mode,
                                   &softmax_alpha,
                                   srcTensorDesc, _input_d,
                                   &softmax_beta,
                                   dstTensorDesc, output.buffer_d));
    
    // Persist output
    if(_output_h == nullptr) _output_h = new float[batch_size * num_features];
    checkCUDA(cudaMemcpy(_output_h, output.buffer_d, sizeof(float) * batch_size * num_features, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(output.buffer_h, output.buffer_d, sizeof(float) * batch_size * num_features, cudaMemcpyDeviceToHost));

    return output;
}

ImageDto Softmax::Backward(ImageDto &softmax_output, int *labels)
{
    int batch_size = softmax_output.batch_size;
    int num_features = softmax_output.num_features;

    // Set dx 
    ImageDto dx = ImageDto(batch_size, num_features);
    dx.buffer_h = (float*)malloc(sizeof(float) * batch_size * num_features);
    checkCUDA(cudaMalloc(&dx.buffer_d, sizeof(float) * batch_size * num_features));

    if(backward_library == CUDNN) {            
        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, num_features, 1, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, num_features, 1, 1));
        // Set dy   
        float *dy_d;
        checkCUDA(cudaMalloc(&dy_d, sizeof(float) * batch_size * num_features));

        if(dy_cal == GPU) {
            int *labels_d = labels;
            checkCUDA(cudaMalloc(&labels_d, sizeof(int) * batch_size * num_features));
            checkCUDA(cudaMemcpy(labels_d, labels, sizeof(int) * batch_size * num_features, cudaMemcpyHostToDevice));

            if(_softmax_algo == CUDNN_SOFTMAX_LOG) {
                // Put y = LogSoftmax. Then, NLLLoss = -t x y -->  dNLLLoss/dy = -t   
                diffNLLLoss<<<batch_size, num_features>>>(batch_size, labels_d, dy_d);   
            }
            else {
                // Put y = Softmax. Then, NLLLoss = -t x log(y) -->  dNLLLoss/dy = -t/y  
                diffNLLoss<<<batch_size, num_features>>>(batch_size, softmax_output.buffer_d, labels_d, dy_d);
            }
            checkCUDA(cudaDeviceSynchronize());
            checkCUDA(cudaFree(labels_d));
        }
        else { ////* CPU Version *////            
            float dy_h[batch_size * num_features];
            if(_softmax_algo == CUDNN_SOFTMAX_LOG){
                // Put y = LogSoftmax. Then, NLLLoss = -t x y -->  dNLLLoss/dy = -t  
                for(int i=0; i<batch_size * num_features; i++){
                    dy_h[i] = labels[i]==1 ? (float)(-1.f / batch_size) : 0.f; //normalize it by batch_size;
                }
            }else{
                // Put y = Softmax. Then, NLLLoss = -t x log(y) -->  dNLLLoss/dy = -t/y  
                for(int i=0; i<batch_size * num_features; i++){
                    dy_h[i] = labels[i]==1 ? (float)(-1.f  / softmax_output.buffer_h[i] / batch_size) : 0.f;   //normalize it by batch_size;
                }
            }
            checkCUDA(cudaMemcpy(dy_d, dy_h, sizeof(float) * (batch_size * num_features), cudaMemcpyHostToDevice));
            free(dy_h);
        }
       
        // Softmax backward
        float softmax_alpha = 1;
        float softmax_beta =  0;
        checkCUDNN(cudnnSoftmaxBackward(cudnnHandle,
                                        _softmax_algo,
                                        _softmax_mode,
                                        &softmax_alpha,
                                        dstTensorDesc, softmax_output.buffer_d,  //Softmax y
                                        dstTensorDesc, dy_d,                     // DL/dy
                                        &softmax_beta,
                                        srcTensorDesc, dx.buffer_d));             // DL/dx
        checkCUDA(cudaFree(dy_d));
    }
    else { //* CUBLAS *//
        // Type Casting of labels from int to float
        float *labels_casted = (float*)malloc(sizeof(float)* num_features * batch_size);
        
        
        for(int i=0; i<num_features*batch_size; i++){
            labels_casted[i] = (float)labels[i];
        }
        checkCUDA(cudaMemcpy(dx.buffer_d, labels_casted, sizeof(float) * (batch_size * num_features), cudaMemcpyHostToDevice));

        float alpha = -1.f;
        float scale = (float)(-1.f / batch_size);

        if(_softmax_algo == CUDNN_SOFTMAX_LOG) {
            takeExponential<<<batch_size, num_features>>>(softmax_output.buffer_d);
        }
        // dx = (softmax_output - target) / batch_size for cross entropy loss                    
        checkCUBLAS(cublasSaxpy(cublasHandle, batch_size*num_features, &alpha, softmax_output.buffer_d, 1, dx.buffer_d, 1));
        checkCUBLAS(cublasSscal_v2(cublasHandle, batch_size*num_features, &scale, dx.buffer_d, 1));
        
        free(labels_casted);
    }
    
    checkCUDA(cudaMemcpy(dx.buffer_h, dx.buffer_d, sizeof(float) * (batch_size * num_features), cudaMemcpyDeviceToHost));

    // free input DTO memory
    free(softmax_output.buffer_h);
    checkCUDA(cudaFree(softmax_output.buffer_d));

    return dx;
}