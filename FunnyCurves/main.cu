#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>

using namespace std;
using namespace cv;

#define MAX_THREADS 32

__constant__ float PI = 3.1415;

enum curves { Gaussian, Line, Circle, Ellipse, ArchimedeanSpiral, Cardioid };

// int __float_as_int_(fvalue)
// https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases
// https://my.oschina.net/hardbone/blog/798552

__global__ void drawLine(float* src, size_t inputPitch, int rows, int cols, float slope, float pitch, float* dst, size_t outputPitch, float thickness)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    float x = (float)(col - cols / 2);
    float y = (float)(rows / 2 - row);

    if (row < rows&&col < cols)
    {
        if (fabsf(y - (slope*x + pitch)) <= thickness)
        {
            float* outputPixel = (float*)((char*)dst +row*outputPitch) + col;
            *outputPixel = 0.0; // make this point of pixel value as black
        }
    }
}

__global__ void drawCircle(float* src, size_t inputPitch, int rows, int cols, float centerX, float centerY, float radius, float* dst, size_t outputPitch, float thickness)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    float x = (float)(col - cols / 2);
    float y = (float)(rows / 2 - row);

    if (row < rows&&col < cols)
    {
        float t = sqrtf(powf(x - centerX, 2) + powf(y - centerY, 2));
        if (t >= radius && t <= radius + thickness)
        {
            float* outputPixel = (float*)((char*)dst + row*outputPitch) + col;
            *outputPixel = 0.0; // make this point of pixel value as black
        }
    }
}

void drawCurves(const Mat & input, float slope, float pitch, Mat & output, curves type, float thickness)
{
    // define blocks size and threads size
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceCount - 1);

    /*
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least,
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */
    dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS);

    size_t inputPitch, outputPitch;
    float* src; float* dst;
    cudaStream_t inputStream, outputStream;
    cudaStreamCreate(&inputStream); cudaStreamCreate(&outputStream);

    cudaMallocPitch(&src, &inputPitch, sizeof(float)*input.cols, input.rows);
    cudaMemcpy2DAsync(src, inputPitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream);

    cudaMallocPitch(&dst, &outputPitch, sizeof(float)*output.cols, output.rows);
    cudaMemcpy2DAsync(dst, outputPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream);

    cudaStreamSynchronize(inputStream); cudaStreamSynchronize(outputStream);

    cudaError_t error;
    switch (type)
    {
    case Gaussian:
        break;
    case Line:
        drawLine <<<blockSize, threadSize>>> (src, inputPitch, input.rows, input.cols, slope, pitch, dst, outputPitch, thickness);
        break;
    case Circle:
        drawCircle <<<blockSize, threadSize >>> (src, inputPitch, input.rows, input.cols, 10, 10, 80, dst, outputPitch, thickness);
        break;
    case Ellipse:
        break;
    default:
        break;
    }
    error = cudaDeviceSynchronize();
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
    Mat white(Size(801, 601), CV_8U, Scalar(255)); // use odd number of size is convenient for computing
    white.convertTo(white, CV_32F);

    Mat result = white.clone();
    float theta = 45.0;
    float slope = tan(theta*3.14 / 180.0);
    float pitch = 30.0;
    float thickness = 5;

    float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);

    drawCurves(white, slope, pitch, result, Circle, thickness);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cout << "time cost on device: " << time << " ms." << endl;

    result.convertTo(result, CV_8U);

    string title = "CUDA";
    namedWindow(title);
    imshow(title, result);

    waitKey(0);

    return 0;
}