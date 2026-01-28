#include <iostream>
#include <opencv2/opencv.hpp>
#include "main.h"
#include "config.h"

using namespace std;
using namespace cv;

void saveImages(unsigned char* data, int height, int width, int dim, const char* folderPath) {
    for (int i = 0; i < dim; ++i) {
        Mat image(height, width, CV_8UC1, data + i * height * width);
        string fileName = string(folderPath) + "/i_" + to_string(i) + "_result.bmp";
        imwrite(fileName, image);
    }
}

void saveArrayToFile(unsigned char* array, int height, int width, int dim, const char* filename) {
    FILE* fp = fopen(filename, "wb"); 
    fwrite(&dim, sizeof(int), 1, fp);
    fwrite(&height, sizeof(int), 1, fp);
    fwrite(&width, sizeof(int), 1, fp);
    fwrite(array, sizeof(unsigned char), dim * height * width, fp);
    fclose(fp);
}


void processImage(const std::string& imagePath) {
    Mat img, img_input;

    // Read image
    img = imread(imagePath, 0);
    int img_W = img.rows;
    int img_H = img.cols;

    // Padding
    int top = (SS + NS - 2) / 2;
    int bottom = (SS + NS - 2) / 2;
    int left = (SS + NS - 2) / 2;
    int right = (SS + NS - 2) / 2;
    int dim_after_paddig = (SS + NS - 2) + depth;
    copyMakeBorder(img, img_input, top, bottom, left, right, BORDER_REFLECT_101);

    // Padded image size
    int W = img_input.rows;
    int H = img_input.cols;

    // Allocate memory
    unsigned char* GPU_input = new unsigned char[W * H * dim_after_paddig];
    float* GPU_result = new float[img_W * img_H * depth];
    unsigned char* result_uc = new unsigned char[img_W * img_H * depth];

    // Write padded input image
    // imwrite("../data/input.bmp", img_input);

    for (int d = 0; d < dim_after_paddig; ++d) { // Loop through dimensions 
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                GPU_input[(d * W * H) + (i * W) + j] = img_input.at<uchar>(i, j);
            }
        }
    }


    // Perform filtering on GPU
    NLMeansProcessor::NL_Means(GPU_input, GPU_result);


    for (int i = 0; i < img_W * img_H * depth; i++) {

        result_uc[i] = (unsigned char)(GPU_result[i]);

    }

    string BinaryPath = "output.bin";
    saveArrayToFile(result_uc, img_W, img_H, depth, BinaryPath.c_str());

    string folderPath = "../ImgResult";
    saveImages(result_uc, img_H, img_W, depth, folderPath.c_str());


    // Free memory
    delete[] GPU_input;
    delete[] GPU_result;
    delete[] result_uc;
}

int main(int argc, char* argv[]) 
{
//    /*
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <image_path>\n";
            return 1;
        }

        std::string imagePath = argv[1];

        processImage(imagePath);

        return 0;
//     //*/
//    processImage("../data/512.bmp");
}
