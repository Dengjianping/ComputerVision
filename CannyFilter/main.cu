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

__constant__ float PI = 3.1415;
__constant__ float sobelKernelXC[3][3] = { { -1.0,0.0,1.0 },{ -2.0,0.0,2.0 },{ -1.0,0.0,1.0 } };
__constant__ float sobelKernelYC[3][3] = { { 1.0,2.0,1.0 },{ 0.0,0.0,0.0 },{ -1.0,-2.0,-1.0 } };

__device__ float twoDimGaussian(int x, int y, float theta)
{
    float coeffient = 1 / (2 * PI*powf(theta, 2));
    float powerIndex = -(powf(x, 2) + powf(y, 2)) / (2 * powf(theta, 2));
    return coeffient*expf(powerIndex);
}

__global__ void cannyFilter(float* src, float* dst, size_t srcPitch, size_t dstPitch, int rows, int cols, int gaussianRadius, float theta, float* gradientX, float* gradientY)
{
    extern __shared__ float gaussianKernel[];
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < rows && col < cols)
    {
        // 1. Firstly, blur image with gaussian kernel
        if (row < 2 * gaussianRadius + 1 && col < 2 * gaussianRadius + 1)
        {
            gaussianKernel[row*(2 * gaussianRadius + 1) + col] = twoDimGaussian(col - gaussianRadius, gaussianRadius - row, theta);
            __syncthreads();
        }
        // convolving
        for (size_t i = 0; i < 2 * gaussianRadius + 1; i++)
        {
            for (size_t j = 0; j < 2 * gaussianRadius + 1; j++)
            {
                // convolving, about how addressing matrix in device, 
                // see this link http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
                float *inputValue = (float *)((char *)src + row*srcPitch) + col;
                float *outputValue = (float *)((char *)dst + (row + i)*dstPitch) + (col + j);                
                *outputValue += (float)(*inputValue) * gaussianKernel[i*(2 * gaussianRadius + 1) + j];
            }
        }
        
        // 2. Secondly, apply sobel filter to get gradient matrix for x direction and y direction
        for (size_t i = 0; i < 3; i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                float* pixelValue = (float*)((char*)output + row*inputPitch) + col;

                float* gxValue = (float*)((char*)gradientX + (row + i)*outputPitch) + (col + j);
                // convolving gx
                *gxValue += sobelKernelXC[i][j]*(*pixelValue);

                float* gyValue = (float*)((char*)gradientY + (row + i)*outputPitch) + (col + j);
                // convolving gy
                *gyValue += sobelKernelYC[i][j]*(*pixelValue);
            }
        }
    }
}

void CannyFilter(Mat & input, Mat & output)
{
    // convert uchar to float for computing conveniently
    input.convertTo(input, CV_32F);
    output.convertTo(output, CV_32F);
    
    float* src; float* dst;
    size_t srcPitch, dstPitch;
    
    cudaStream_t srcStream, dstStream;
    cudaStreamCreate(&srcStream); cudaStreamCreate(&dstStream);
    
    // allocate pitch for src and dst
    cudaMallocPitch(&src, &srcPitch, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&dst, &dstPitch, sizeof(float)*output.cols, output.rows);
    
    // copy data to device
    cudaMemcpy2DAsync(src, srcPitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, srcStream);
    cudaMemcpy2DAsync(dst, dstPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, dstStream);
    
    // hold here to wait data copy complete
    cudaStreamSynchronize(srcStream); cudaStreamSynchronize(dstStream);
    
    
    float* gradientX; float* gradientY;
    size_t gxPitch, gyPitch;
    cudaMallocPitch(&gradientX, &gxPitch, sizeof(float)*output.cols, output.rows);
    cudaMallocPitch(&gradientY, &gyPitch, sizeof(float)*output.cols, output.rows);
    
    cudaStream_t gxStream, gyStream;
    cudaStreamCreate(&gxStream); cudaStreamCreate(&gyStream);
    cudaMemcpy2DAsync(gradientX, gxPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, gxStream);
    cudaMemcpy2DAsync(gradientY, gyPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, gyStream);
    cudaStreamSynchronize(gxStream); cudaStreamSynchronize(gyStream);
}

int main()
{
    string path = "type-c.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F);
    float alpha = 2;
    Mat result(Size(img.cols*alpha, img.rows*alpha), CV_32F, Scalar(0));
    
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    
    cudaEventRecord(start);
    //resizeImage(img, result);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    
    float time;
    cudaEventElapsedTime(&time, start, end);
    cout << "time cost co GPU: " << time << " ms." << endl;

    string title = "CUDA";
    namedWindow(title);
    result.convertTo(result, CV_8U);
    imshow(title, result);
    waitKey(0);

    return 0;
}