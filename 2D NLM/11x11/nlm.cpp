#include <iostream>
#include <opencv2/opencv.hpp>
#include "main.h"
using namespace cv;

void processImage(const std::string& imagePath, int sr_size, int nb_size) {
    Mat img, img_input;

    // Read image
    img = imread(imagePath, 0);
    int img_W = img.rows;
    int img_H = img.cols;

    // Padding
    int top = (sr_size + nb_size - 2) / 2;
    int bottom = (sr_size + nb_size - 2) / 2;
    int left = (sr_size + nb_size - 2) / 2;
    int right = (sr_size + nb_size - 2) / 2;
    copyMakeBorder(img, img_input, top, bottom, left, right, BORDER_REFLECT_101);

    // Padded image size
    int W = img_input.rows;
    int H = img_input.cols;

    // Allocate memory
    unsigned char* GPU_input = new unsigned char[W * H];
    float* GPU_result = new float[img_W * img_H];

    // Write padded input image
    imwrite("../data/input.bmp", img_input);

    // Copy data to GPU input
    memcpy(GPU_input, img_input.data, W * H * sizeof(unsigned char));

    // Perform filtering on GPU
    NLMeansProcessor::NL_Means(GPU_input, GPU_result, W, H, sr_size, nb_size, img_H, img_W);

    // Create output image
    Mat temp(img_W, img_H, CV_32FC1, GPU_result);
    Mat output = temp.clone();

    // Write output image
    imwrite("../data/output.bmp", output);

    // Free memory
    delete[] GPU_input;
    delete[] GPU_result;
}

int main(int argc, char* argv[]) {
 ///*
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <search_size> <neighbor_size>\n";
        return 1;
    }

    std::string imagePath = argv[1];
    int searchSize = std::stoi(argv[2]);
    int neighborSize = std::stoi(argv[3]);

    processImage(imagePath, searchSize, neighborSize);

    return 0;
 //*/   
  //   processImage("../data/2048.bmp",21,11);
}


