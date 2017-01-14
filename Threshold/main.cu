#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_THREADS 32

using namespace std;
using namespace cv;

__global__ void threshold(uchar *input, size_t inputPitch, int rows, int cols, uchar *output, uchar thresholdValue) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < rows&&col < cols) {
        uchar *pixelValue = (uchar*)((char*)input + row*inputPitch) + col;
        uchar *outputPixelValue = (uchar*)((char*)output + row*inputPitch) + col;
        if (*pixelValue < thresholdValue) {
            *outputPixelValue = 0;
        }
        else {
            *outputPixelValue = 255;
        }
    }
}

void thresholdImage(const Mat & input, Mat & output, uchar threholdValue) {
    output = Mat(input.size(), CV_8U, Scalar(0));

    uchar *d_input, *d_output;
    size_t inputPitch, outputPitch;
    cudaMallocPitch(&d_input, &inputPitch, sizeof(uchar)*input.cols, input.rows);
    cudaMallocPitch(&d_output, &outputPitch, sizeof(uchar)*output.cols, output.rows);

    cudaStream_t inputCopy, outputCopy;
    cudaStreamCreate(&inputCopy); cudaStreamCreate(&outputCopy);

    cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, inputCopy);
    cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, outputCopy);

    cudaStreamSynchronize(inputCopy); cudaStreamSynchronize(outputCopy);

    dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS);

    threshold<<<blockSize, threadSize>>> (d_input, inputPitch, input.rows, input.cols, d_output, threholdValue);

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        cout << cudaGetErrorString(error) << endl;
    }

    cudaMemcpy2D(output.data, output.cols * sizeof(uchar), d_output, inputPitch, output.cols * sizeof(uchar), output.rows, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(inputCopy); cudaStreamDestroy(outputCopy);
    cudaFree(d_input); cudaFree(d_output);
}

int main() {
    string path = "type-c.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);
    Mat img1 = imread(path);

    Mat result;

    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);
    thresholdImage(img, result, 50);
    cudaEventRecord(end);
    cudaEventSynchronize(start); cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start); cudaEventDestroy(end);
    cout << "time cost on cpu: " << time << " ms." << endl;

    Mat th;
    double cpuStart = (double)getTickCount();
    threshold(img, th, 50, 255, img.type());
    double cpuEnd = (double)getTickCount();
    double cpuTime = (cpuEnd - cpuStart) / getTickFrequency();
    cout << "time cost on cpu: " << cpuTime * 1000 << " ms." << endl;

    string title = "CUDA";
    namedWindow(title);
    imshow(title, result);
    imshow("CPU", th);
    waitKey(0);
    imwrite("threshold.jpg", result);

    return 0;
}