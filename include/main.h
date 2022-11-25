#pragma once

#include <string>
#include <vector>

class MNIST
{
    public:
        MNIST();
        
        int n_images;
        int n_labels;
        int n_rows;
        int n_cols; 
    
        std::vector<float> image_data; //n_images * n_rows * n_cols
        std::vector<int> label_data; //n_images * n_rows * n_cols

        void load_data(bool is_train, std::string file_path);
        void get_next_batch(int batch_size, float* imgs, int* labels);
        void print_imgs(int batch_size, float* batch_imgs, int* batch_labels);
        void reset();

        int reverse_int(int i);
        
    private:
        int n_batch;
        int batch_num_pixels;
        int total_num_pixels;

        std::string img_file;
        std::string label_file;
};