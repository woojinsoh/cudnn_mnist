#include <iostream>
#include <cudnn.h>
// #include <cudnn_frontend.h>

#include "fused_layer.h"
#include "utils.h"
#include "fp16_dev.h"

Conv2dBiasReLU::Conv2dBiasReLU(std::string layer_name, int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation):_layer_name(layer_name), _in_channels(in_channels), _out_channels(out_channels), _kernel_size(kernel_size), _stride(stride), _padding(padding), _dilation(dilation){
    initWeightBias(1, _layer_name, _in_channels, _out_channels, _kernel_size); // Initialize weight and bias
}

int Conv2dBiasReLU::getFwdConvDilatedFilterDim(int filterDim, int dilation){
    return (dilation * (filterDim - 1) + 1);
}

int Conv2dBiasReLU::getFwdConvPaddedImageDim(int tensorDim, int pad){
    return tensorDim + (2 * pad);
}

int Conv2dBiasReLU::getFwdConvOutputDim(int tensorDim, int pad, int filterDim, int stride, int dilation){
    int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return p;
}

void Conv2dBiasReLU::generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    /* For INT8x4 and INT8x32 we still compute standard strides here to input
       into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref. */
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1] = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

// cuDNN backend APIs
void Conv2dBiasReLU::setCudnnBackendFwdDescriptors(){
    int64_t pad[] = {_padding, _padding};
    int64_t stride[] = {_stride, _stride};
    int64_t dilation[] = {_dilation, _dilation};

    //////////////////////////////
    /*1. Set tensor descriptors */
    //////////////////////////////                   
    cudnnDataType_t dtype = CUDNN_DATA_HALF; //TensorCore is mandatory for runtime fusion. So, FP16 might be required unless Ampere.
    int64_t alignment = 16; //16B alighment is needed to run a tensor core engine

    //x
    int64_t xDim[] = {in_n, in_c, in_h, in_w};
    int64_t xStr[4];
    int64_t xUid = 'x';
    if(info_flag) printf(">> Convolution input shape(NHWC):: %ld x %ld x %ld x %ld\n", in_n, in_h, in_w, in_c);
    generateStrides(xDim, xStr, 4, CUDNN_TENSOR_NHWC); 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, xDim));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, xStr));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &xUid));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(xDesc));

    //w
    int64_t wDim[] = {_out_channels, _in_channels, _kernel_size, _kernel_size};
    int64_t wStr[4]; 
    int64_t wUid = 'w';
    generateStrides(wDim, wStr, 4, CUDNN_TENSOR_NHWC);  // filter layout for NHWC: [_out_channels, _kernel_size, _kernel_size, _in_channels]
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &wDesc));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, wDim));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, wStr));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &wUid));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(wDesc));

    //b
    int64_t bDim[] = {1, _out_channels, 1, 1};
    int64_t bStr[4];
    int64_t bUid = 'b';
    generateStrides(bDim, bStr, 4, CUDNN_TENSOR_NHWC);
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &bDesc));
    checkCUDNN(cudnnBackendSetAttribute(bDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(bDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, bDim));
    checkCUDNN(cudnnBackendSetAttribute(bDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, bStr));
    checkCUDNN(cudnnBackendSetAttribute(bDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &bUid));
    checkCUDNN(cudnnBackendSetAttribute(bDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(bDesc));

    //z intermediate result(conv)
    out_n = in_n;
    out_c = _out_channels;
    out_h = getFwdConvOutputDim(xDim[2], pad[0], wDim[2], stride[0], dilation[0]);
    out_w = getFwdConvOutputDim(xDim[3], pad[1], wDim[3], stride[1], dilation[1]);
    int64_t zDim[] = {out_n, out_c, out_h, out_w};
    int64_t zStr[4];
    int64_t zUid = 'z';
    bool isVirtual = true;
    if(info_flag) printf(">> Convolution output shape(NHWC):: %ld x %ld x %ld x %ld\n", out_n, out_h, out_w, out_c);
    generateStrides(zDim, zStr, 4, CUDNN_TENSOR_NHWC);
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &zDesc));
    checkCUDNN(cudnnBackendSetAttribute(zDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(zDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, zDim));
    checkCUDNN(cudnnBackendSetAttribute(zDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, zStr));
    checkCUDNN(cudnnBackendSetAttribute(zDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &zUid));
    checkCUDNN(cudnnBackendSetAttribute(zDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendSetAttribute(zDesc, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &isVirtual)); //intermediate result that doesn't need to be traced
    checkCUDNN(cudnnBackendFinalize(zDesc));
    
    //a intermediate result(conv + bias)
    int64_t aDim[] = {out_n, out_c, out_h, out_w};
    int64_t aStr[4];
    int64_t aUid = 'a';
    if(info_flag) printf(">> BiasAdd output shape(NHWC):: %ld x %ld x %ld x %ld\n", out_n, out_h, out_w, out_c);
    generateStrides(aDim, aStr, 4, CUDNN_TENSOR_NHWC);
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &aDesc));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, aDim));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, aStr));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &aUid));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(aDesc));

    //y final result(conv + bias + relu)
    int64_t yDim[] = {out_n, out_c, out_h, out_w};
    int64_t yStr[4];
    int64_t yUid = 'y';
    if(info_flag) printf(">> Activation output shape(NHWC):: %ld x %ld x %ld x %ld\n", out_n, out_h, out_w, out_c);
    generateStrides(yDim, yStr, 4, CUDNN_TENSOR_NHWC);   
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc));
    checkCUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, yDim));
    checkCUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, yStr));
    checkCUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &yUid));
    checkCUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(yDesc));

    /////////////////////////////////////////////
    /* 2. Set computatioal operator descriptor */
    /////////////////////////////////////////////
    // Convolution descriptor
    int64_t nbDims = 2;
    int64_t convtype = CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION; //  cross correlation is only supported in runtime fusion
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convDesc));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &nbDims));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &convtype));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, nbDims, dilation));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, nbDims, stride));
    checkCUDNN(cudnnBackendFinalize(convDesc));

    // Pointwise add descriptor(add bias)
    cudnnPointwiseMode_t pwMode = CUDNN_POINTWISE_ADD;
    cudnnDataType_t biastype = CUDNN_DATA_FLOAT;
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pointWiseAdd));
    checkCUDNN(cudnnBackendSetAttribute(pointWiseAdd, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &pwMode));
    checkCUDNN(cudnnBackendSetAttribute(pointWiseAdd, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &biastype));
    checkCUDNN(cudnnBackendFinalize(pointWiseAdd));

    // Activation(ReLU) descriptor
    cudnnPointwiseMode_t actMode= CUDNN_POINTWISE_RELU_FWD;
    cudnnDataType_t relutype = CUDNN_DATA_FLOAT;
    float clip = 0.0;
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &reluDesc));
    checkCUDNN(cudnnBackendSetAttribute(reluDesc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &actMode));
    checkCUDNN(cudnnBackendSetAttribute(reluDesc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &relutype));
    checkCUDNN(cudnnBackendSetAttribute(reluDesc, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, CUDNN_TYPE_FLOAT, 1, &clip));
    checkCUDNN(cudnnBackendFinalize(reluDesc));

    /////////////////////////////////////
    /* 3. Forward operation descriptor */
    /////////////////////////////////////
    // The operational flow is:: z=conv(x,w) -> a=add(z,bias) -> y=ReLU(a)
    // Convolution node
    float conv_alpha = 1;
    float conv_beta = 0;
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &convFprop));    
    checkCUDNN(cudnnBackendSetAttribute(convFprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &conv_alpha));
    checkCUDNN(cudnnBackendSetAttribute(convFprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &conv_beta));
    checkCUDNN(cudnnBackendFinalize(convFprop));
    
    // Bias node
    //input to this is z and output is "a" as shown in the operation graph
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &biasFprop));
    checkCUDNN(cudnnBackendSetAttribute(biasFprop, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zDesc));
    checkCUDNN(cudnnBackendSetAttribute(biasFprop, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bDesc));
    checkCUDNN(cudnnBackendSetAttribute(biasFprop, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &aDesc));
    checkCUDNN(cudnnBackendSetAttribute(biasFprop, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pointWiseAdd));
    checkCUDNN(cudnnBackendFinalize(biasFprop));

    // Activation(ReLU) node
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &reluFprop));    
    checkCUDNN(cudnnBackendSetAttribute(reluFprop, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &aDesc));
    checkCUDNN(cudnnBackendSetAttribute(reluFprop, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc));
    checkCUDNN(cudnnBackendSetAttribute(reluFprop, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &reluDesc));
    checkCUDNN(cudnnBackendFinalize(reluFprop));

    ///////////////////////////////////////
    /* 4. Set Operation graph descriptor */
    ///////////////////////////////////////
    cudnnBackendDescriptor_t ops[] = {convFprop, biasFprop, reluFprop};   // Runtime Fusion
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));
    checkCUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 3, ops));
    checkCUDNN(cudnnBackendSetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(opGraph)); // This might have an error regarding tensor size

    //////////////////////////////
    /* 5. Set Engine Descriptor */
    //////////////////////////////
    int64_t globalCount = -1;
    int64_t gIdx = 0;
    checkCUDNN(cudnnBackendGetAttribute(opGraph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1, NULL, &globalCount));
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
    checkCUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraph));
    checkCUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gIdx));
    checkCUDNN(cudnnBackendFinalize(engine));

    //////////////////////////////////
    /* 6. Set Engine cfg Descriptor */
    //////////////////////////////////
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engCfg));
    checkCUDNN(cudnnBackendSetAttribute(engCfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));
    checkCUDNN(cudnnBackendFinalize(engCfg));

    /////////////////////////////////////////////////////////////////////////
    /* 7. Set Execution plan descritpor, secure workspace size to allocate */
    ///////////////////////////////////////////////////////////////////////// 
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan);
    checkCUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engCfg));
    checkCUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(plan));

    checkCUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &_fwd_workspace_size));

    ////////////////////////////////////
    /* 8. Set variant pack descriptor */
    ////////////////////////////////////
    // Allocate the workspace
    checkCUDA(cudaMalloc(&_fwd_workspace_d, _fwd_workspace_size * sizeof(float)));

    // Set NHWC layout output 
    if(_fusion_output_fp16_d == nullptr) checkCUDA(cudaMalloc(&_fusion_output_fp16_d, sizeof(half1) * out_n * out_c * out_h * out_w));
    if(_after_conv_bias_fp16_d == nullptr) checkCUDA(cudaMalloc(&_after_conv_bias_fp16_d, sizeof(half1) * out_n * out_c * out_h * out_w));

    // Set Variant pack
    /* CUDNN: This is the ordering of the pointers. 
       Here, _weight_d, _bias_d are considered as NHWC layout.*/

    void *devPtrs[] = {_input_fp16_d, _after_conv_bias_fp16_d, _fusion_output_fp16_d, _weight_fp16_d, _bias_fp16_d}; 
    int64_t uids[] = {'x', 'a', 'y', 'w', 'b'};  
    checkCUDNN((cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varPack)));
    checkCUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 5, devPtrs));
    checkCUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 5, uids));
    checkCUDNN(cudnnBackendSetAttribute(varPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &_fwd_workspace_d));
    checkCUDNN(cudnnBackendFinalize(varPack));
}

void Conv2dBiasReLU::setCudnnBackendBwdDescriptors(){
    int64_t pad[] = {_padding, _padding};
    int64_t stride[] = {_stride, _stride};
    int64_t dilation[] = {_dilation, _dilation};

    //////////////////////////////
    /*1. Set tensor descriptors */
    ////////////////////////////// 
    /* NHWC layout seems to be mandatory for backend APIs*/
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    int64_t alignment = 16;
    
    //x: might not need to be defined again as it's done when forwarding.
    int64_t xDim[] = {in_n, in_c, in_h, in_w};
    int64_t xStr[4];
    int64_t xUid = 103;
    generateStrides(xDim, xStr, 4, CUDNN_TENSOR_NHWC); 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, xDim));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, xStr));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &xUid));
    checkCUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(xDesc));

    //a intermediate result(conv + bias): might not need to be defined again as it's done when forwarding.
    int64_t aDim[] = {out_n, out_c, out_h, out_w};
    int64_t aStr[4];
    int64_t aUid = 104;
    generateStrides(aDim, aStr, 4, CUDNN_TENSOR_NHWC);
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &aDesc));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, aDim));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, aStr));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &aUid));
    checkCUDNN(cudnnBackendSetAttribute(aDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(aDesc));

    //w: might not need to be defined again as it's done when forwarding.
    int64_t wDim[] = {_out_channels, _in_channels, _kernel_size, _kernel_size};
    int64_t wStr[4]; 
    int64_t wUid = 'w';
    generateStrides(wDim, wStr, 4, CUDNN_TENSOR_NHWC);  // filter layout for NHWC: [_out_channels, _kernel_size, _kernel_size, _in_channels]
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &wDesc));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, wDim));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, wStr));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &wUid));
    checkCUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(wDesc));

    //dy
    int64_t dyDim[] = {out_n, out_c, out_h, out_w};
    int64_t dyStr[4];
    int64_t dyUid = 101;    
    generateStrides(dyDim, dyStr, 4, CUDNN_TENSOR_NHWC); 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &dyDesc));
    checkCUDNN(cudnnBackendSetAttribute(dyDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(dyDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dyDim));
    checkCUDNN(cudnnBackendSetAttribute(dyDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, dyStr));
    checkCUDNN(cudnnBackendSetAttribute(dyDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &dyUid));
    checkCUDNN(cudnnBackendSetAttribute(dyDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(dyDesc));

    //da (activation backward output)
    int64_t daDim[] = {out_n, out_c, out_h, out_w};
    int64_t daStr[4];
    int64_t daUid = 'da';    
    bool isVirtual = false;
    generateStrides(daDim, daStr, 4, CUDNN_TENSOR_NHWC); 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &daDesc));
    checkCUDNN(cudnnBackendSetAttribute(daDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(daDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, daDim));
    checkCUDNN(cudnnBackendSetAttribute(daDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, daStr));
    checkCUDNN(cudnnBackendSetAttribute(daDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &daUid));
    checkCUDNN(cudnnBackendSetAttribute(daDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendSetAttribute(daDesc, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &isVirtual));
    checkCUDNN(cudnnBackendFinalize(daDesc));
    
    //db (bias backward output)
    int64_t dbDim[] = {1, out_c, 1, 1};
    int64_t dbStr[4];
    int64_t dbUid = 'db';    
    generateStrides(dbDim, dbStr, 4, CUDNN_TENSOR_NHWC); 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &dbDesc));
    checkCUDNN(cudnnBackendSetAttribute(dbDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(dbDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dbDim));
    checkCUDNN(cudnnBackendSetAttribute(dbDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, dbStr));
    checkCUDNN(cudnnBackendSetAttribute(dbDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &dbUid));
    checkCUDNN(cudnnBackendSetAttribute(dbDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(dbDesc));

    //dW (conv backward filter output)
    int64_t dwDim[] = {_out_channels, _in_channels, _kernel_size, _kernel_size};
    int64_t dwStr[4];
    int64_t dwUid = 102;    
    generateStrides(dwDim, dwStr, 4, CUDNN_TENSOR_NHWC); 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &dwDesc));
    checkCUDNN(cudnnBackendSetAttribute(dwDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(dwDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dwDim));
    checkCUDNN(cudnnBackendSetAttribute(dwDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, dwStr));
    checkCUDNN(cudnnBackendSetAttribute(dwDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &dwUid));
    checkCUDNN(cudnnBackendSetAttribute(dwDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(dwDesc));

    //dx (conv backward data output)
    int64_t dxDim[] = {in_n, in_c, in_h, in_w};
    int64_t dxStr[4];
    int64_t dxUid = 'dx';    
    generateStrides(dxDim, dxStr, 4, CUDNN_TENSOR_NHWC); 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &dxDesc));
    checkCUDNN(cudnnBackendSetAttribute(dxDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
    checkCUDNN(cudnnBackendSetAttribute(dxDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dxDim));
    checkCUDNN(cudnnBackendSetAttribute(dxDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, dxStr));
    checkCUDNN(cudnnBackendSetAttribute(dxDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &dxUid));
    checkCUDNN(cudnnBackendSetAttribute(dxDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkCUDNN(cudnnBackendFinalize(dxDesc));

    //////////////////////////////////////////////
    /* 2. Set computational operator descriptor */
    //////////////////////////////////////////////
    float alpha = 1;
    float beta = 0;
    // Activation(ReLU) descriptor
    cudnnPointwiseMode_t actMode= CUDNN_POINTWISE_RELU_BWD;
    cudnnDataType_t acttype = CUDNN_DATA_FLOAT;
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &reluDesc));
    checkCUDNN(cudnnBackendSetAttribute(reluDesc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &actMode));
    checkCUDNN(cudnnBackendSetAttribute(reluDesc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &acttype));    
    checkCUDNN(cudnnBackendFinalize(reluDesc));

    // Convolution descriptor
    int64_t nbDims = 2; 
    int64_t convtype = CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION; //  cross correlation is only supported in runtime fusion
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convDesc));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &nbDims));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &convtype));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, nbDims, dilation));
    checkCUDNN(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, nbDims, stride));
    checkCUDNN(cudnnBackendFinalize(convDesc));

    // bias add descriptor
    cudnnReduceTensorOp_t rdMode = CUDNN_REDUCE_TENSOR_ADD;
    cudnnDataType_t biastype = CUDNN_DATA_FLOAT;
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_REDUCTION_DESCRIPTOR, &biasDesc));
    checkCUDNN(cudnnBackendSetAttribute(biasDesc, CUDNN_ATTR_REDUCTION_OPERATOR, CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &rdMode));
    checkCUDNN(cudnnBackendSetAttribute(biasDesc, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &biastype));
    checkCUDNN(cudnnBackendFinalize(biasDesc));

    //////////////////////////////////////
    /* 3. Backward operation descriptor */
    //////////////////////////////////////
    // gradient ReLU
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &reluBprop));    
    checkCUDNN(cudnnBackendSetAttribute(reluBprop, CUDNN_ATTR_OPERATION_POINTWISE_DYDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dyDesc));
    checkCUDNN(cudnnBackendSetAttribute(reluBprop, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &aDesc));
    checkCUDNN(cudnnBackendSetAttribute(reluBprop, CUDNN_ATTR_OPERATION_POINTWISE_DXDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &daDesc));
    checkCUDNN(cudnnBackendSetAttribute(reluBprop, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &reluDesc));
    checkCUDNN(cudnnBackendFinalize(reluBprop));

    // gradient Bias node
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR, &biasBprop));
    checkCUDNN(cudnnBackendSetAttribute(biasBprop, CUDNN_ATTR_OPERATION_REDUCTION_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &daDesc));
    checkCUDNN(cudnnBackendSetAttribute(biasBprop, CUDNN_ATTR_OPERATION_REDUCTION_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dbDesc));
    checkCUDNN(cudnnBackendSetAttribute(biasBprop, CUDNN_ATTR_OPERATION_REDUCTION_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &biasDesc));
    // checkCUDNN(cudnnBackendSetAttribute(biasBprop, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &alpha));
    // checkCUDNN(cudnnBackendSetAttribute(biasBprop, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, CUDNN_TYPE_FLOAT, 1, &beta));
    checkCUDNN(cudnnBackendFinalize(biasBprop));

    // gradient Conv node
    //Data
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, &convDataBprop));    
    checkCUDNN(cudnnBackendSetAttribute(convDataBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &daDesc));
    checkCUDNN(cudnnBackendSetAttribute(convDataBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc));
    checkCUDNN(cudnnBackendSetAttribute(convDataBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dxDesc));
    checkCUDNN(cudnnBackendSetAttribute(convDataBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convDesc));
    checkCUDNN(cudnnBackendSetAttribute(convDataBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    checkCUDNN(cudnnBackendSetAttribute(convDataBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
    checkCUDNN(cudnnBackendFinalize(convDataBprop));

    //Filter
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR, &convFilterBprop));    
    checkCUDNN(cudnnBackendSetAttribute(convFilterBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &daDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFilterBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFilterBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dwDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFilterBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convDesc));
    checkCUDNN(cudnnBackendSetAttribute(convFilterBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    checkCUDNN(cudnnBackendSetAttribute(convFilterBprop, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
    checkCUDNN(cudnnBackendFinalize(convFilterBprop));

    ///////////////////////////////////////
    /* 4. Set Operation graph descriptor */
    ///////////////////////////////////////    
    /* Define single operation graph respectively. The pattern used in this example doesn't support runtime fusion.
       https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#runtime-fusion-engine */
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &dReluGraph));
    checkCUDNN(cudnnBackendSetAttribute(dReluGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &reluBprop));
    checkCUDNN(cudnnBackendSetAttribute(dReluGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dReluGraph));

    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &dBiasGraph));
    checkCUDNN(cudnnBackendSetAttribute(dBiasGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &biasBprop));
    checkCUDNN(cudnnBackendSetAttribute(dBiasGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dBiasGraph));

    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &dConvDataGraph));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convDataBprop));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dConvDataGraph));

    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &dConvFilterGraph));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterGraph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convFilterBprop));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterGraph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dConvFilterGraph));

    //////////////////////////////
    /* 5. Set Engine Descriptor */
    //////////////////////////////
    int64_t globalCount = -1;
    int64_t gIdx = 0;
    checkCUDNN(cudnnBackendGetAttribute(dReluGraph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1, NULL, &globalCount));
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &dReluEngine));
    checkCUDNN(cudnnBackendSetAttribute(dReluEngine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dReluGraph));
    checkCUDNN(cudnnBackendSetAttribute(dReluEngine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gIdx));
    checkCUDNN(cudnnBackendFinalize(dReluEngine));

    checkCUDNN(cudnnBackendGetAttribute(dBiasGraph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1, NULL, &globalCount));
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &dBiasEngine));
    checkCUDNN(cudnnBackendSetAttribute(dBiasEngine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dBiasGraph));
    checkCUDNN(cudnnBackendSetAttribute(dBiasEngine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gIdx));
    checkCUDNN(cudnnBackendFinalize(dBiasEngine));

    checkCUDNN(cudnnBackendGetAttribute(dConvDataGraph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1, NULL, &globalCount));
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &dConvDataEngine));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataEngine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dConvDataGraph));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataEngine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gIdx));
    checkCUDNN(cudnnBackendFinalize(dConvDataEngine));

    checkCUDNN(cudnnBackendGetAttribute(dConvFilterGraph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1, NULL, &globalCount));
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &dConvFilterEngine));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterEngine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dConvFilterGraph));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterEngine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gIdx));
    checkCUDNN(cudnnBackendFinalize(dConvFilterEngine));
    
    //////////////////////////////////
    /* 6. Set Engine cfg Descriptor */
    //////////////////////////////////
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &dReluEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dReluEngCfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dReluEngine));
    checkCUDNN(cudnnBackendFinalize(dReluEngCfg));

    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &dBiasEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dBiasEngCfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dBiasEngine));
    checkCUDNN(cudnnBackendFinalize(dBiasEngCfg));
    
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &dConvDataEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataEngCfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dConvDataEngine));
    checkCUDNN(cudnnBackendFinalize(dConvDataEngCfg));
    
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &dConvFilterEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterEngCfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dConvFilterEngine));
    checkCUDNN(cudnnBackendFinalize(dConvFilterEngCfg));

    /////////////////////////////////////////////////////////////////////////
    /* 7. Set Execution plan descritpor, obtain workspace size to allocate */
    ///////////////////////////////////////////////////////////////////////// 
    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &dReluPlan));
    checkCUDNN(cudnnBackendSetAttribute(dReluPlan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dReluEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dReluPlan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dReluPlan)); 

    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &dBiasPlan));
    checkCUDNN(cudnnBackendSetAttribute(dBiasPlan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dBiasEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dBiasPlan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dBiasPlan)); 

    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &dConvDataPlan));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataPlan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dConvDataEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataPlan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dConvDataPlan));    

    checkCUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &dConvFilterPlan));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterPlan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &dConvFilterEngCfg));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterPlan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnnHandle));
    checkCUDNN(cudnnBackendFinalize(dConvFilterPlan)); 
        
    ////////////////////////////////////
    /* 8. Set variant pack descriptor */
    ////////////////////////////////////       
    // Allocate the workspace
    checkCUDNN(cudnnBackendGetAttribute(dReluPlan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &dRelu_workspace_size));
    checkCUDA(cudaMalloc(&_dRelu_workspace_d, dRelu_workspace_size * sizeof(float)));

    checkCUDNN(cudnnBackendGetAttribute(dBiasPlan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &dBias_workspace_size));
    checkCUDA(cudaMalloc(&_dBias_workspace_d, dBias_workspace_size * sizeof(float)));

    checkCUDNN(cudnnBackendGetAttribute(dConvDataPlan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &dConvData_workspace_size));
    checkCUDA(cudaMalloc(&_dConvData_workspace_d, dConvData_workspace_size * sizeof(float)));

    checkCUDNN(cudnnBackendGetAttribute(dConvFilterPlan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &dConvFilterworkspace_size));
    checkCUDA(cudaMalloc(&_dConvFilter_workspace_d, dConvFilterworkspace_size * sizeof(float))); 

    // Set Variant Packs
    void *dReluDataPtrs[] = {_dy_d, _after_conv_bias_fp16_d, _da_d};
    int64_t dReluUids[] = {dyUid, aUid, daUid};
    checkCUDNN((cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &dReluVarPack)));
    checkCUDNN(cudnnBackendSetAttribute(dReluVarPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, dReluDataPtrs));
    checkCUDNN(cudnnBackendSetAttribute(dReluVarPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, dReluUids));
    checkCUDNN(cudnnBackendSetAttribute(dReluVarPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &_dRelu_workspace_d));
    checkCUDNN(cudnnBackendFinalize(dReluVarPack));

    void *dBiasDataPtrs[] = {_da_d, _grad_bias_d};
    int64_t dBiasUids[] = {daUid, dbUid};
    checkCUDNN((cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &dBiasVarPack)));
    checkCUDNN(cudnnBackendSetAttribute(dBiasVarPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 2, dBiasDataPtrs));
    checkCUDNN(cudnnBackendSetAttribute(dBiasVarPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 2, dBiasUids));
    checkCUDNN(cudnnBackendSetAttribute(dBiasVarPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &_dBias_workspace_d));
    checkCUDNN(cudnnBackendFinalize(dBiasVarPack));

    void *dConvDataDataPtrs[] = {_da_d, _weight_d, _dx_d};
    int64_t dConvDataUids[] = {daUid, wUid, dxUid};
    checkCUDNN((cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &dConvDataVarPack)));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataVarPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, dConvDataDataPtrs));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataVarPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, dConvDataUids));
    checkCUDNN(cudnnBackendSetAttribute(dConvDataVarPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &_dConvData_workspace_d));
    checkCUDNN(cudnnBackendFinalize(dConvDataVarPack));

    void *dConvFilterDataPtrs[] = {_da_d, _input_d, _grad_weight_d};
    int64_t dConvFilterUids[] = {daUid, xUid, dwUid};
    checkCUDNN((cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &dConvFilterVarPack)));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterVarPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, dConvFilterDataPtrs));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterVarPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, dConvFilterUids));
    checkCUDNN(cudnnBackendSetAttribute(dConvFilterVarPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &_dConvFilter_workspace_d));
    checkCUDNN(cudnnBackendFinalize(dConvFilterVarPack));   
}

ImageDto Conv2dBiasReLU::Forward(ImageDto &input){
    num_features = input.num_features;
    in_n = input.batch_size;
    in_c = input.num_channels;
    in_h = input.height;
    in_w = input.weight;

    if(info_flag) std::cout << "Backend API fused kernel FWD:: [Layer Name:: " << _layer_name << "]" << std::endl;  
    /* Convert input layout from NCHW to NHWC. NHWC layout is mandatory for runtime fusion. 
       cuDNN Backend/Front APIs seem to require explicit transformation.You can use cudnnTransformTensor() API for transformation */
    int64_t permute[3] = {0,2,1}; // [N, C, H*W] --> [N, H*W, C]
    if(permute_d==nullptr) {
        checkCUDA(cudaMalloc(&permute_d, sizeof(int64_t) * 3));
        checkCUDA(cudaMemcpy(permute_d, &permute, sizeof(int64_t) * 3, cudaMemcpyHostToDevice));
    }

    int64_t input_dims3d[3] = {in_n, in_c, in_h*in_w};
    if(input_dims3d_d == nullptr) {
        checkCUDA(cudaMalloc(&input_dims3d_d, sizeof(int64_t) * 3));
        checkCUDA(cudaMemcpy(input_dims3d_d, &input_dims3d, sizeof(int64_t) * 3, cudaMemcpyHostToDevice));
    }

    if(_input_h == nullptr) _input_h = (float*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
    if(_input_fp16_h == nullptr) _input_fp16_h = (half1*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
    if(_input_d == nullptr) checkCUDA(cudaMalloc(&_input_d, sizeof(float) * in_n * in_c * in_h * in_w));    
    if(_input_fp16_d == nullptr) checkCUDA(cudaMalloc(&_input_fp16_d, sizeof(half1) * in_n * in_c * in_h * in_w));

    // Convert from NCHW into NHWC
    int64_t nthreads = 512;
    int64_t nblocks = ceil((double)(in_n*in_c*in_h*in_w)/nthreads); 
    permute3dTensorsKernel<<<nblocks, nthreads>>>(input.buffer_d, input_dims3d_d, permute_d, _input_d);
    checkCUDA(cudaDeviceSynchronize());
    if(info_flag) std::cout << ">> Input layout conversion:: from NCHW to NHWC." << std::endl;  

    /* FP32 to FP 16 */
    // Copy input from FP32 to FP16
    gpu_float2half_rn(in_n*in_c*in_h*in_w, _input_d, _input_fp16_d);
    checkCUDA(cudaDeviceSynchronize());
    if(info_flag) std::cout << ">> Copy inputs:: from FP32 to FP16." << std::endl;  

    // Copy weights from FP32 to FP16'
    if(_weight_fp16_d == nullptr) checkCUDA(cudaMalloc(&_weight_fp16_d, sizeof(half1) * _out_channels * _in_channels * _kernel_size * _kernel_size));
    gpu_float2half_rn(_out_channels * _in_channels * _kernel_size * _kernel_size, _weight_d, _weight_fp16_d);
    checkCUDA(cudaDeviceSynchronize());

    // Copy biases from FP32 to FP16'
    if(_bias_fp16_d == nullptr) checkCUDA(cudaMalloc(&_bias_fp16_d, sizeof(half1) * _out_channels));
    gpu_float2half_rn(_out_channels, _weight_d, _weight_fp16_d);
    checkCUDA(cudaDeviceSynchronize());
    if(info_flag) std::cout << ">> Copy weights/biases:: from FP32 to FP16." << std::endl;  

    // Persist input
    checkCUDA(cudaMemcpy(_input_h, _input_d, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(_input_fp16_h, _input_fp16_d, sizeof(half1) * in_n * in_c * in_h * in_w, cudaMemcpyDeviceToHost));
        
    // Free input DTO memory
    free(input.buffer_h);
    checkCUDA(cudaFree(input.buffer_d));

    // Set Backend API FWD descriptors
    if(backend_fwd_flag){
        setCudnnBackendFwdDescriptors();
        backend_fwd_flag = 0;
    }    

    // Execution
    checkCUDNN(cudnnBackendExecute(cudnnHandle, plan, varPack));
    if(info_flag) std::cout << ">> Conv/Bias/Activation Done:: with FP16\n" << std::endl;  


    // Copy output back from FP16 to FP32
    if(_output_d == nullptr) checkCUDA(cudaMalloc(&(_output_d), sizeof(float) * (out_n * out_c * out_h * out_w))); 
    gpu_half2float(out_n * out_c * out_h * out_w, _fusion_output_fp16_d, _output_d);
    checkCUDA(cudaDeviceSynchronize());
    if(info_flag) std::cout << ">> Copy outputs:: from FP16 to FP32" << std::endl;  


    // Set output DTO
    ImageDto output = ImageDto(out_n, out_c, out_h, out_w);
    output.buffer_h = (float*)malloc(sizeof(float) * (out_n * out_c * out_h * out_w));
    checkCUDA(cudaMalloc(&(output.buffer_d), sizeof(float) * (out_n * out_c * out_h * out_w)));   

    // Convert output layout from NHWC to NCHW
    int64_t output_dims3d[3] = {out_n, out_h*out_w, out_c};
    if(output_dims3d_d == nullptr) {
        checkCUDA(cudaMalloc(&output_dims3d_d, sizeof(int64_t) * 3));
        checkCUDA(cudaMemcpy(output_dims3d_d, &output_dims3d, sizeof(int64_t) * 3, cudaMemcpyHostToDevice));
    }
    
    nblocks = ceil((double)(out_n*out_c*out_h*out_w)/nthreads); 
    permute3dTensorsKernel<<<nblocks, nthreads>>>(_output_d, output_dims3d_d, permute_d, output.buffer_d);
    checkCUDA(cudaDeviceSynchronize());
    if(info_flag) std::cout << ">> Output layout conversion:: from NHWC to NCHW" << std::endl;  
    if(info_flag) printf(">> Activation output shape(NCHW):: %ld x %ld x %ld x %ld\n\n", out_n, out_c, out_h, out_w);

    // Persist outputs
    if(_output_h == nullptr) _output_h = new float[out_n * out_c * out_h * out_w];    
    checkCUDA(cudaMemcpy(_output_h, output.buffer_d, sizeof(float) * (out_n * out_c * out_h * out_w), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(output.buffer_h, output.buffer_d, sizeof(float) * (out_n * out_c * out_h * out_w), cudaMemcpyDeviceToHost));

    // for(int i=0; i<out_n * out_c * out_h * out_w; i++){
    //     printf("%f ", output.buffer_h[i]);
    // }

    return output;
}

ImageDto Conv2dBiasReLU::Backward(ImageDto &dy, int *labels){
    // Set dy
    if(_dy_d == nullptr) checkCUDA(cudaMalloc(&_dy_d, sizeof(float) * (out_n * out_c * out_h * out_w)));   

    // Set dW
    if(_grad_weight_h == nullptr) _grad_weight_h = (float*)malloc(sizeof(float) * (_out_channels * _kernel_size * _kernel_size * _in_channels));
    if(_grad_weight_d == nullptr) checkCUDA(cudaMalloc(&_grad_weight_d, sizeof(float) * (_out_channels * _kernel_size * _kernel_size * _in_channels)));

    
    // Set db
    if(_grad_bias_h == nullptr) _grad_bias_h = (float*)malloc(sizeof(float) *_out_channels);
    if(_grad_bias_d == nullptr) checkCUDA(cudaMalloc(&_grad_bias_d, sizeof(float) * _out_channels));
    
    // Set dx
    if(_dx_d == nullptr) checkCUDA(cudaMalloc(&_dx_d, sizeof(float)* in_n*in_c*in_h*in_w));

    // Set da
    if(_da_d == nullptr) checkCUDA(cudaMalloc(&_da_d, sizeof(float)* out_n*out_c*out_h*out_w));
    
    // Convert dy layout from NCHW to NHWC. NHWC layout seems to be mandatory for backward computation with backend APIs.
    int64_t dy_dims3d[] = {out_n, out_c, out_h*out_w};
    if(dy_dims3d_d == nullptr) {
        checkCUDA(cudaMalloc(&dy_dims3d_d, sizeof(int64_t) * 3));    
        checkCUDA(cudaMemcpy(dy_dims3d_d, &dy_dims3d, sizeof(int64_t) * 3, cudaMemcpyHostToDevice));
    }
    int64_t nthreads = 256;
    int64_t nblocks = ceil((double)(out_n*out_c*out_h*out_w)/nthreads); 
    permute3dTensorsKernel<<<nblocks, nthreads>>>(dy.buffer_d, dy_dims3d_d, permute_d, _dy_d);
    checkCUDA(cudaDeviceSynchronize());
    
    // Set Backend API BWD descriptors
    if(backend_bwd_flag){
        setCudnnBackendBwdDescriptors();
        backend_bwd_flag = 0;
    }    

    // Execution
    checkCUDNN(cudnnBackendExecute(cudnnHandle, dReluPlan, dReluVarPack));
    checkCUDNN(cudnnBackendExecute(cudnnHandle, dBiasPlan, dBiasVarPack));
    checkCUDNN(cudnnBackendExecute(cudnnHandle, dConvFilterPlan, dConvFilterVarPack));
    checkCUDNN(cudnnBackendExecute(cudnnHandle, dConvDataPlan, dConvDataVarPack));

    // Set output DTO
    ImageDto dx = ImageDto(in_n, in_c, in_h, in_w);
    dx.buffer_h = (float*)malloc(sizeof(float) * in_n * in_c * in_h * in_w);
    checkCUDA(cudaMalloc(&(dx.buffer_d), sizeof(float) * in_n * in_c * in_h * in_w));

    // Convert back dx layout from NHWC to NCHW
    int64_t dx_dims3d[] = {in_n, in_h*in_w, in_c};
    if(dx_dims3d_d == nullptr) {
        checkCUDA(cudaMalloc(&dx_dims3d_d, sizeof(int64_t) * 3));    
        checkCUDA(cudaMemcpy(dx_dims3d_d, &dx_dims3d, sizeof(int64_t) * 3, cudaMemcpyHostToDevice));
    }
    
    nblocks = ceil((double)(in_n*in_c*in_h*in_w)/nthreads); 
    permute3dTensorsKernel<<<nblocks, nthreads>>>(_dx_d, dx_dims3d_d, permute_d, dx.buffer_d);
    checkCUDA(cudaDeviceSynchronize());
    
    // Persist outputs
    checkCUDA(cudaMemcpy(_grad_bias_h, _grad_bias_d, sizeof(float) * _out_channels, cudaMemcpyDeviceToHost));    
    checkCUDA(cudaMemcpy(_grad_weight_h, _grad_weight_d, sizeof(float) * _out_channels * _in_channels * _kernel_size * _kernel_size, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(dx.buffer_h, dx.buffer_d, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyDeviceToHost));
    
    // Free input DTO memory
    free(dy.buffer_h);
    checkCUDA(cudaFree(dy.buffer_d));

    return dx;
}