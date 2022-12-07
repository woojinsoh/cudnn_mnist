#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <iomanip>

#include "main.h"
#include "layer.h"
#include "fused_layer.h"
#include "utils.h"
#include "model.h"
#include "loss.h"

int MNIST::reverse_int(int i)
/* Big-Endian reading */
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNIST::reset(){
    image_data.clear();
    label_data.clear();
    n_images = 0;
    n_labels = 0;
    n_rows = 0;
    n_cols = 0;
    n_batch = 0;
    img_file ="";
    label_file="";
}

void MNIST::load_data(bool is_train, std::string data_dir) {
    int temp_magic_number = 0;
    unsigned char pixel;
    unsigned char label;

    // get file path
    if(is_train){
        img_file = "train-images-idx3-ubyte";
        label_file = "train-labels-idx1-ubyte";
    } else {
        img_file = "t10k-images-idx3-ubyte";
        label_file = "t10k-labels-idx1-ubyte";        
    }
    std::string img_fpath = data_dir + "/" + img_file;
    std::string label_fpath = data_dir + "/" + label_file;
    
    std::cout << "Load MNIST images from " << img_fpath << std::endl;
    
    // loading image data
    std::ifstream img_file_stream(img_fpath);

    img_file_stream.read((char*)&temp_magic_number, sizeof(temp_magic_number));
    img_file_stream.read((char*)&n_images, sizeof(n_images));
    n_images = reverse_int(n_images);
    printf(">> Num Images:: %d\n", n_images);

    img_file_stream.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);
    img_file_stream.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);
    printf(">> Image Size:: %d x %d\n", n_rows, n_cols);
        
    total_num_pixels = n_images * n_rows * n_cols;
    for(int i=0; i<total_num_pixels; i++) {
        img_file_stream.read((char*)&pixel, sizeof(pixel));
        image_data.push_back((float)pixel/255);
    }
    img_file_stream.close();
    std::cout << "Load MNIST labels from " << label_fpath << std::endl;
    // loading label data
    std::ifstream label_file_stream(label_fpath);
    label_file_stream.read((char*)&temp_magic_number, sizeof(temp_magic_number));
    label_file_stream.read((char*)&n_labels, sizeof(n_labels));
    n_labels = reverse_int(n_labels);
    printf(">> Num Labels:: %d\n", n_labels);

        
    for(int i=0; i<n_images; i++) {
        label_file_stream.read((char*)&label, sizeof(label));
        label_data.push_back((int)label);
    }
    label_file_stream.close();
    std::cout << "MNIST dataset is successfully loaded.\n" << std::endl;
}

void MNIST::print_imgs(int batch_size, float* batch_imgs, int* batch_labels){
    printf("%dst batch\n", n_batch);
    for(int k=0;k<batch_size; k++){
        for(int i=0;i<n_rows;i++){
            for(int j=0;j<n_cols;j++){
                if(batch_imgs[k*n_rows*n_cols + i*n_rows+j]==0) printf("%d", (int)batch_imgs[k*n_rows*n_cols +i*n_rows+j]);            
                else printf("%d", 1);            
            }
            printf("\n");
        }
    }
    for(int i=0;i<batch_size; i++){
        printf("%d ", batch_labels[i]);
    };
    printf("\n");
}

void MNIST::get_next_batch(int batch_size, float* batch_imgs, int* batch_labels)
{
    batch_num_pixels = batch_size * n_rows * n_cols;

    if((n_batch + 1) * batch_num_pixels - 1 < total_num_pixels){
        copy(image_data.begin() + n_batch * batch_num_pixels, image_data.begin() + (n_batch + 1) * batch_num_pixels, batch_imgs);
        copy(label_data.begin() + n_batch * batch_size, label_data.begin() + (n_batch + 1) * batch_size, batch_labels);
        n_batch++;
        // print_imgs(batch_size, batch_imgs, batch_labels);
    }
    else{
        // printf("All data is exhausted. Start loading from the beginning again.\n");
        if(n_batch * batch_num_pixels == total_num_pixels){            
            copy(image_data.begin() + n_batch * batch_num_pixels, image_data.begin() + (n_batch + 1) * batch_num_pixels, batch_imgs);
            copy(label_data.begin() + n_batch * batch_size, label_data.begin() + (n_batch + 1) * batch_size, batch_labels);
        }
        else{
            copy(image_data.begin() + n_batch * batch_num_pixels, image_data.end(), batch_imgs);
            copy(label_data.begin() + n_batch * batch_size, label_data.end(), batch_labels);
        }
        n_batch = 0;
    }
};

MNIST::MNIST():n_batch(0), batch_num_pixels(0), total_num_pixels(0){};

int main(int argc, char *argv[])
{
    // MNIST info
    int batch_size = 128;
    int num_channels = 1;
    int num_classes = 10;

    // Memory for labels
    int *labels_h = (int*)malloc(sizeof(int) * batch_size);            
    int *labels_d = nullptr;
    checkCUDA(cudaMalloc(&labels_d, sizeof(int) * batch_size));
    
    int *onehot_labels_h = (int*)malloc(sizeof(int) * batch_size * 10);
    int *onehot_labels_d = nullptr;
    std::fill_n(onehot_labels_h, batch_size * num_classes, 0);   // initiatilzation to 0
    checkCUDA(cudaMalloc(&onehot_labels_d, sizeof(int) * batch_size * num_classes));

    // Loss, accuracy
    float *loss;
    float *accuracy;
    checkCUDA(cudaMallocManaged(&loss, sizeof(float)));
    checkCUDA(cudaMallocManaged(&accuracy, sizeof(float)));
    cudnnSoftmaxAlgorithm_t softmax_algo = CUDNN_SOFTMAX_LOG;

    // Load data
    std::cout << "[Data Loading]" << std::endl;
    std::string data_dir = "../dataset";
    MNIST mnist = MNIST();
    mnist.load_data(1, data_dir);
    ImageDto data = ImageDto(batch_size, num_channels, mnist.n_rows, mnist.n_cols);
    
    /* Define models.
       Note that layout should be power of 4,8,16, ...*/
    std::cout << "[Weight Initialization]" << std::endl;
    Model model = Model();
    model.addLayers(new Conv2D("conv1_and_bias", 1, 32, 5, 1, 0, 1));
    model.addLayers(new Activation("act1_relu", CUDNN_ACTIVATION_RELU));
    model.addLayers(new Pooling("pool1_max", 2, 2, 0, CUDNN_POOLING_MAX));    

    //  model.addLayers(new Conv2D("conv2_and_bias", 32, 64, 5, 1, 0, 1));           // with CUDNN
    //  model.addLayers(new Activation("act2_relu", CUDNN_ACTIVATION_RELU));         // with CUDNN
    model.addLayers(new Conv2dBiasReLU("conv2_and_bias_and_relu", 32, 64, 5, 1, 0, 1));   // with CUDNN Backend APIs
    model.addLayers(new Pooling("pool2_max", 2, 2, 0, CUDNN_POOLING_MAX));
    
    model.addLayers(new Dense("dense1_and_bias", 1024, 100));
    model.addLayers(new Dense("dense2_and_bias", 100, 10));
    
    model.addLayers(new Softmax("softmax", softmax_algo, CUDNN_SOFTMAX_MODE_CHANNEL));
    std::cout << "\n[Model Architecture]" << std::endl;
    
    // Training loop
    int num_steps = 2000;    
    for(int i=0;i<num_steps;i++){
        data.buffer_h = (float*)malloc(sizeof(float) * (batch_size * num_channels * mnist.n_rows * mnist.n_cols));  // imgs
        mnist.get_next_batch(data.batch_size, data.buffer_h, labels_h);        
        checkCUDA(cudaMemcpy(labels_d, labels_h, sizeof(int) * batch_size, cudaMemcpyHostToDevice));

        oneHotEncoding(batch_size, num_classes, labels_h, onehot_labels_h);
        checkCUDA(cudaMemcpy(onehot_labels_d, onehot_labels_h, sizeof(int) * batch_size * num_classes, cudaMemcpyHostToDevice));
        
        ImageDto forward_output = model.Forward(data);
        
        if(i==0) {
            std::cout << "[Start training]" << std::endl;
            std::cout << "Num iterations:: " << num_steps << std::endl;
            std::cout << "Batch size:: " << batch_size << std::endl;
        }
        if(i % 100 == 0){
            *loss = model.Loss(forward_output, onehot_labels_d, softmax_algo);
            *accuracy = model.Accuracy(forward_output, labels_d, num_classes); //need to check this part. error
            std::cout << "At iteration " << std::right << std::setw(4) << i << \
                         ",    Expected Loss:: " << std::fixed << std::setprecision(4) << *loss << \
                         "    Accuracy to the current batch:: " << std::fixed << std::setprecision(4) << *accuracy << \
            std::endl;
        }
        ImageDto backward_output = model.Backward(forward_output, onehot_labels_h);   
        model.Update(0.001);
        
        free(backward_output.buffer_h);
        checkCUDA(cudaFree(backward_output.buffer_d));
    }
    
    // Load inference dataset
    std::cout << "" << std::endl;
    std::cout << "[Load test dataset]" << std::endl;
    mnist.reset();
    mnist.load_data(0, data_dir);

    // inference accuracy
    float total_acc = 0;
    float *infer_acc_per_batch;
    checkCUDA(cudaMallocManaged(&infer_acc_per_batch, sizeof(float) * mnist.n_images/batch_size));
    
    // Inference loop
    std::cout << "[Start inferencing]" << std::endl;
    int num_infer_steps = mnist.n_images/batch_size;
    std::cout << num_infer_steps * batch_size << " images are used for inference." << std::endl;
    for(int i=0; i<num_infer_steps; i++){
        data.buffer_h = (float*)malloc(sizeof(float) * (batch_size * num_channels * mnist.n_rows * mnist.n_cols));  // imgs
        mnist.get_next_batch(batch_size, data.buffer_h, labels_h);        
        checkCUDA(cudaMemcpy(labels_d, labels_h, sizeof(int) * batch_size, cudaMemcpyHostToDevice));

        ImageDto infer_output = model.Forward(data);        
        infer_acc_per_batch[i] = model.Accuracy(infer_output, labels_d, num_classes);
        total_acc = total_acc + infer_acc_per_batch[i] * batch_size;
    }
    total_acc = total_acc / (batch_size * num_infer_steps);
    std::cout << "Total inference accuracy:: " << total_acc << std::endl;

    std::cout << "" << std::endl;
    std::cout << "[Finished]" << std::endl;
    
    return 0;
}
