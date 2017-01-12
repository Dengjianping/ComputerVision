#include <iostream>
#include <string>
#include <stdio.h>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

enum Mode { full, valid, same };

#define MAX_THREADS 32

__constant__ float laplace[3][3] = { {0,1,0},{1,-4,1},{0,1,0} };

__device__ void convolve2D(float *input, int rows, int cols, size_t inputPitch, float *output, size_t outputPitch) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < rows + 3 && col < cols + 3) {
        if (row < rows&&col < cols) {
            // convolving
            for (size_t i = 0; i < 3; i++)
                for (size_t j = 0; j < 3; j++) {
                    float *inputPixelValue = (float*)((char*)input + row*inputPitch) + col;
                    float *outputpixelValue = (float*)((char*)output + (row + i)*outputPitch) + (col + j);
                    *outputpixelValue += laplace[i][j] * (*inputPixelValue);
                }
        }
    }
}

__global__ void laplaceTransform(float *input, int rows, int cols, size_t inputPitch, float *output, size_t outputPitch) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < rows + 3 && col < cols + 3) {
        if (row < rows && col < cols) {
            // convolving
            for (size_t i = 0; i < 3; i++)
                for (size_t j = 0; j < 3; j++) {
                    float *inputPixelValue = (float*)((char*)input + row*inputPitch) + col;
                    float *outputpixelValue = (float*)((char*)output + (row + i)*outputPitch) + (col + j);
                    *outputpixelValue += laplace[i][j] * (*inputPixelValue);
                }
        }
    }
    //convolve2D(input, rows, cols, inputPitch, output, outputPitch);
}

void laplaceTransform(Mat & input, Mat & output) {
    output = Mat(Size(input.cols + 2, input.rows + 2), CV_32F, Scalar(0));

    cout << output.at<float>(0, 0) << endl;

    int channelsCount = input.channels();
    switch (channelsCount) {
    case 1:
    {
        input.convertTo(input, CV_32F);
        // only one channel
        float *d_input, *d_output;
        size_t inputPitch, outputPitch;
        cudaMallocPitch(&d_input, &inputPitch, sizeof(float)*input.cols, input.rows);
        cudaMallocPitch(&d_output, &outputPitch, sizeof(float)*output.cols, output.rows);

        cudaStream_t inputCopyStream, outputCopyStream;
        cudaStreamCreate(&inputCopyStream); cudaStreamCreate(&outputCopyStream);

        cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, inputCopyStream);
        cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, outputCopyStream);

        cudaStreamSynchronize(inputCopyStream); cudaStreamSynchronize(outputCopyStream);

        dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
        dim3 threadSize(MAX_THREADS, MAX_THREADS);

        laplaceTransform<<<blockSize, threadSize>>> (d_input, input.rows, input.cols, inputPitch, d_output, outputPitch);

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cout << cudaGetErrorString(error) << endl;
        }

        cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, outputPitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost);

        cudaStreamDestroy(inputCopyStream); cudaStreamDestroy(outputCopyStream);
        cudaFree(d_input); cudaFree(d_output);

        break;
    }
    default:
        break;
    }
}

int main() {
    string path = "type-c.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);
    cout << img.channels() << endl;
    Mat result;

    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);
    laplaceTransform(img, result);
    cudaEventRecord(end);
    cudaEventSynchronize(start); cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    cout << "time cost on device: " << time << " ms" << endl;

    string title = "CUDA";
    namedWindow(title);

    result.convertTo(result, CV_8U);
    imshow(title, result);
    img.convertTo(img, CV_8U);

    double cpuStart = (double)getTickCount();
    Laplacian(img, img, img.depth());
    double cpuEnd = (double)getTickCount();
    double cpuTime = (cpuEnd - cpuStart) / getTickFrequency();
    cout << "time cost on cpu: " << cpuTime * 1000 << " ms." << endl;

    imshow("CPU", img);
    waitKey(0);

    return 0;
}