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

#define MAX_THREADS 32

using namespace std;
using namespace cv;


enum curves { Gaussian, Line, Circle, ArchimedeanSpiral, Cardioid }; // kinds of curves, definitely there're others interesting curves
__constant__ float PI = 3.1415;


// a line equation like y = k * x + b, so the pixels meet this equation
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

// a circle equation like (x-a)**2 + (y-b)**2 = radius**2, so the pixels meet this equation
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

__device__ float gaussianEquation(,float x, float symmetry, float stdError)
{
    float coeffient = 1 / (sqrtf(2*PI)*stdError);
    float powerIndex = powf(x-symmetry, 2)/(2*powf(stdError,2));
    return coeffient*expf(-powerIndex);
}

// 1-dim gaussian equation like (1/(sqrt(2*PI))*stdError)*exp(-(x-symmetry)**2/(2*(stdError**2)))
__global__ void drawGaussian(float* src, size_t inputPitch, int rows, int cols, float symmetry, float stdError, float* dst, size_t outputPitch, float thickness)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    float x = (float)(col - cols / 2);
    float y = (float)(rows / 2 - row);

    if (row < rows&&col < cols)
    {
        if (fabsf(y - gaussianEquation(x, symmetry, stdError) <=thickness)
        {
            float* outputPixel = (float*)((char*)dst + row*outputPitch) + col;
            *outputPixel = 0.0; // make this point of pixel value as black
        }
    }
}

/*  
    change cartesian coordinate to polor coordinate, example point M(x, y), target P(radius, theta), radius = sqrt(x*x + y*y), theta = arctan(y / x), and 
    archimedean spiral equation like this: radius = a + b*theta, just let the point of pixel meets the equation
*/
__global__ void drawSpiral(float* src, size_t inputPitch, int rows, int cols, float a, float b, float* dst, size_t outputPitch, float thickness)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    float x = (float)(col - cols / 2);
    float y = (float)(rows / 2 - row);

    if (row < rows&&col < cols)
    {
        float radius = sqrtf(x * x+y * y);
        float theta = atanf(y / x);
        if (fabsf(radius - (a+theta*b) <=thickness)
        {
            float* outputPixel = (float*)((char*)dst + row*outputPitch) + col;
            *outputPixel = 0.0; // make this point of pixel value as black
        }
    }
}

/*  
    change cartesian coordinate to polor coordinate, example point M(x, y), target P(radius, theta), radius = sqrt(x*x + y*y), theta = arctan(y / x), 
    and archimedean spiral equation like this: radius = a*(1 - cos(theta)) for vertical direction, radius = a*(1 - sin(theta)) for horizonal direction, 
    just let the point of pixel meets the equation.
*/
__global__ void drawHeart(float* src, size_t inputPitch, int rows, int cols, float a, float* dst, size_t outputPitch, float thickness)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    float x = (float)(col - cols / 2);
    float y = (float)(rows / 2 - row);

    if (row < rows&&col < cols)
    {
        float radius = sqrtf(x * x+y * y);
        float theta = atanf(y / x);
        if (fabsf(radius - a*(1 - cosf(theta)) <=thickness)
        {
            float* outputPixel = (float*)((char*)dst + row*outputPitch) + col;
            *outputPixel = 0.0; // make this point of pixel value as black
        }
    }
}

void drawCurves(const Mat & input, Mat & output, curves type, float thickness)
{
    // define blocks size and threads size
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceCount - 1);

    /*
    my sample image size is 801 * 601, so we need 801 * 601 threads to process this image on device at least,
    each block can contain 1024 threads at most in my device, so ,I can define block size as x = 801 / 32 = 26, y = 801 / 32 = 19
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
        float symmetry = 0.0;
        float stdError = 0.5;
        drawGaussian <<<blockSize, threadSize>>> (src, inputPitch, input.rows, input.cols, symmetry, stdError, dst, outputPitch, thickness);
        break;
    case Line:
        float theta = 45.0;
        float slope = tan(theta*3.14 / 180.0);
        float pitch = 30.0;
        drawLine <<<blockSize, threadSize>>> (src, inputPitch, input.rows, input.cols, slope, pitch, dst, outputPitch, thickness);
        break;
    case Circle:
        float centerX = 20.0;
        float centerY = 50.0;
        float radius = 80.0;
        drawCircle <<<blockSize, threadSize >>> (src, inputPitch, input.rows, input.cols, centerX, centerY, radius, dst, outputPitch, thickness);
        break;
    case ArchimedeanSpiral:
        float a = 20.0;
        float b = 30.0;
        drawSpiral <<<blockSize, threadSize>>> (src, inputPitch, input.rows, input.cols, a, b, dst, outputPitch, thickness);
        break;
    case Cardioid:
        float a = 10.0;
        drawHeart <<<blockSize, threadSize>>> (src, inputPitch, input.rows, input.cols, a, dst, outputPitch, thickness);
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

    Mat result = white.clone(); // deeply copy
    
    float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);
    
    float thickness = 5;
    drawCurves(white, result, Gaussian, thickness);
    drawCurves(white, result, Line, thickness);
    drawCurves(white, result, Circle, thickness);
    drawCurves(white, result, ArchimedeanSpiral, thickness);
    drawCurves(white, result, Cardioid, thickness);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cout << "time cost on device: " << time << " ms." << endl;
    cudaEventDestroy(start); cudaEventDestroy(end);

    result.convertTo(result, CV_8U);

    string title = "CUDA";
    namedWindow(title);
    imshow(title, result);
    waitKey(0);

    return 0;
}