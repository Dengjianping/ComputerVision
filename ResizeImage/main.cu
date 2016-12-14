#include <stdio.h>
#include <iostream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

#define MAX_THREADS 32

using namespace std;
using namespace cv;

__global__ void enlarge(float* src, size_t inputPitch, int rows, int cols, float* dst, size_t outputPitch, float rowRatio, float colRatio)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < rows&&col < cols)
    {
        // get 4 coordinators points back
        /*float* q11 = (float*)((char*)src + ((int)(row*rowRatio))*inputPitch) + (int)(col*colRatio);
        float* q12 = (float*)((char*)src + (((int)(row*rowRatio) + 1))*inputPitch) + (int)(col*colRatio);
        float* q21 = (float*)((char*)src + ((int)(row*rowRatio))*inputPitch) + (int)(col*colRatio) + 1;
        float* q22 = (float*)((char*)src + (((int)(row*rowRatio) + 1))*inputPitch) + (int)(col*colRatio) + 1;*/

        float* q11 = (float*)((char*)src + ((int)(row*colRatio))*inputPitch) + (int)(col*rowRatio);
        float* q12 = (float*)((char*)src + (((int)(row*colRatio) + 1))*inputPitch) + (int)(col*rowRatio);
        float* q21 = (float*)((char*)src + ((int)(row*colRatio))*inputPitch) + (int)(col*rowRatio) + 1;
        float* q22 = (float*)((char*)src + (((int)(row*colRatio) + 1))*inputPitch) + (int)(col*rowRatio) + 1;

        // Bilinear Interpolation
        float* outputPixel = (float*)((char*)dst + row*outputPitch) + col;
        *outputPixel = (1 - rowRatio)*(1 - colRatio)*(*q11) + (1 - rowRatio)*colRatio*(*q12) + rowRatio*(1 - colRatio)*(*q21) + rowRatio*colRatio*(*q22);
    }
}

void resizeImage(const Mat & input, Mat & output)
{
    float rowRatio = (float)input.rows / (float)output.rows;
    float colRatio = (float)input.cols / (float)output.cols;
    // define block size and thread size
    dim3 blockSize(output.cols / MAX_THREADS + 1, output.rows / MAX_THREADS + 1);
    cout << blockSize.x * blockSize.y *1024 << endl;
    cout << input.rows*input.cols << endl;
    dim3 threadSize(MAX_THREADS, MAX_THREADS);

    cudaStream_t inputStream, outputStream;
    cudaStreamCreate(&inputStream); cudaStreamCreate(&outputStream);

    size_t inputPitch, outputPitch;
    float* src; float* dst;
    cudaMallocPitch(&src, &inputPitch, sizeof(float)*input.cols, input.rows);
    cudaMemcpy2DAsync(src, inputPitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream);

    cudaMallocPitch(&dst, &outputPitch, sizeof(float)*output.cols, output.rows);
    cudaMemcpy2DAsync(dst, outputPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream);

    cudaStreamSynchronize(inputStream); cudaStreamSynchronize(outputStream);

    enlarge<<<blockSize, threadSize >>>(src, inputPitch, output.rows, output.cols, dst, outputPitch, rowRatio, colRatio);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cout << cudaGetErrorString(error) << endl;
    }
    cudaMemcpy2D(output.data, sizeof(float)*output.cols, dst, outputPitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost);

    // resource releasing
    cudaStreamDestroy(inputStream); cudaStreamDestroy(outputStream);
    cudaFree(src); cudaFree(dst);
}

int main()
{
    string path = "type-c.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F);
    float alpha = 2;
    Mat result(Size(img.cols*alpha, img.rows*alpha), CV_32F, Scalar(0));

    resizeImage(img, result);

    string title = "CUDA";
    namedWindow(title);
    result.convertTo(result, CV_8U);
    imshow(title, result);
    waitKey(0);

    return 0;
}