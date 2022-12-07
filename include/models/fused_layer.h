#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda.h>

#include "layer.h"
#include "fp16_dev.h"

class Conv2dBiasReLU: public Layer
{
    private:
        /* Tensor and Operator */
        cudnnBackendDescriptor_t convDesc, pointWiseAdd, biasDesc, reluDesc;
        cudnnBackendDescriptor_t xDesc, wDesc, yDesc, bDesc, zDesc, aDesc;
        cudnnBackendDescriptor_t dxDesc, dwDesc, dyDesc, dbDesc, dzDesc, daDesc;
        
        /* Forward */
        cudnnBackendDescriptor_t convFprop, biasFprop, reluFprop, opGraph, engine, engCfg;
        cudnnBackendDescriptor_t varPack, plan;

        /* Backward */
        cudnnBackendDescriptor_t convDataBprop, convFilterBprop, biasBprop, reluBprop;
        cudnnBackendDescriptor_t dReluGraph, dBiasGraph, dConvDataGraph, dConvFilterGraph;
        cudnnBackendDescriptor_t dReluEngine, dBiasEngine, dConvDataEngine, dConvFilterEngine;
        cudnnBackendDescriptor_t dReluEngCfg, dBiasEngCfg, dConvDataEngCfg, dConvFilterEngCfg;
        cudnnBackendDescriptor_t dReluPlan, dBiasPlan, dConvDataPlan, dConvFilterPlan;
        cudnnBackendDescriptor_t dReluVarPack, dBiasVarPack, dConvDataVarPack, dConvFilterVarPack;
    
        /* Layer meta */
        std::string _layer_name;
        int64_t _in_channels;
        int64_t _out_channels;
        int64_t _kernel_size;
        int64_t _stride;
        int64_t _padding;
        int64_t _dilation;    

        int64_t in_n, in_c, in_h, in_w;
        int64_t out_n, out_c, out_h, out_w;
        int64_t num_features;

        /* Forward Algo workspace */
        float _fwd_workspace_size = 0;
        void *_fwd_workspace_d = nullptr;

        /* Backward Algo workspace */
        float dRelu_workspace_size, dBias_workspace_size, dConvData_workspace_size, dConvFilterworkspace_size;
        void *_dRelu_workspace_d = nullptr;
        void *_dBias_workspace_d = nullptr;
        void *_dConvData_workspace_d = nullptr;
        void *_dConvFilter_workspace_d = nullptr;

        /* For NHWC - NCHW conversion */
        int64_t* input_dims3d_d = nullptr;
        int64_t* output_dims3d_d = nullptr;
        int64_t* dy_dims3d_d = nullptr;
        int64_t* dx_dims3d_d = nullptr;
        int64_t* permute_d = nullptr;
        void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat);

        void setCudnnBackendFwdDescriptors();
        void setCudnnBackendBwdDescriptors();
        bool backend_fwd_flag = 1;
        bool backend_bwd_flag = 1;

        /* Get output_dims info */
        int getFwdConvDilatedFilterDim(int filterDim, int dilation);
        int getFwdConvPaddedImageDim(int tensorDim, int pad);
        int getFwdConvOutputDim(int tensorDim, int pad, int filterDim, int stride, int dilation);
        
        /* fp16 copy tensors for mixed precision */
        half1 *_input_fp16_h = nullptr;
        half1 *_input_fp16_d = nullptr;

        half1 *_weight_fp16_d = nullptr;
        half1 *_bias_fp16_d = nullptr;
        half1 *_grad_weight_fp16_d = nullptr;
        half1 *_grad_bias_fp16_d = nullptr;

        
        /* Forward results - NHWC layout */
        half1 *_after_conv_bias_fp16_d = nullptr;  //activation input used for BP
        half1 *_fusion_output_fp16_d = nullptr;


        /* Backward results - NHWC layout */
        float *_dx_d = nullptr;
        float *_da_d = nullptr;
        float *_dy_d = nullptr;
        
    public:
        Conv2dBiasReLU(std::string layer_name, int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, int dilation=1);
        virtual ImageDto Forward(ImageDto &data);
        virtual ImageDto Backward(ImageDto &data, int *labels);

};