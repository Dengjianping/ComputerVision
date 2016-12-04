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

__constant__ float PI = 3.1415;

enum curves { Gaussian, Line, Circle, Ellipse };

__global__ void drawLine(float* src, size_t inputPitch, int rows, int cols, float theta, float pitch, float* dst, size_t outputPitch)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    float slope = tanh(theta*PI / 180.0);

    float x = (float)(col - cols / 2);
    float y = (float)(rows / 2 - row);

    if (row < rows&&col < cols)
    {
        if (y - (slope*x + pitch) <= 10.0&&y - (slope*x + pitch)>=-10.0)
        {
            float* outputPixel = (float*)((char*)dst +row*outputPitch) + col;
            *outputPixel = 0.0; // make this point of pixel value as black
        }
    }
}

__global__ void drawCircle()
{}

void drawCurves(const Mat & input, float theta, float pitch, Mat & output, curves type)
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
    int blockCount = (int)(input.rows * input.cols / prop.maxThreadsPerBlock) + 1;
    blockCount = (int)(sqrt(blockCount)) + 1;
    dim3 blockSize(blockCount, blockCount);
    dim3 threadSize(32, 32);

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
        drawLine <<<blockSize, threadSize>>> (src, inputPitch, input.rows, input.cols, theta, pitch, dst, outputPitch);
        break;
    case Circle:
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

    Mat result = white;
    float theta = 30;
    float pitch = 30;

    float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);

    drawCurves(white, theta, pitch, result, Line);

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