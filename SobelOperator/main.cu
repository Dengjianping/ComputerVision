#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

using namespace std; 
using namespace cv;

#define MAX_THREADS 32

__constant__ float sobelKernelXC[3][3] = { { -1.0,0.0,1.0 },{ -2.0,0.0,2.0 },{ -1.0,0.0,1.0 } };
__constant__ float sobelKernelYC[3][3] = { { 1.0,2.0,1.0 },{ 0.0,0.0,0.0 },{ -1.0,-2.0,-1.0 } };

__global__ void sobelOperator(float* input, int rows, int cols, size_t inputPitch, float* gx, float* gy, float* output, size_t outputPitch)
{
    // use share memory to accelerate computing
    __shared__ float sobelKernelX[3][3];
    __shared__ float sobelKernelY[3][3];

    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < 3 && col < 3)
    {
        sobelKernelX[row][col] = sobelKernelXC[row][col];
        sobelKernelY[row][col] = sobelKernelYC[row][col];
    }

    if (row < rows + 3 && col < cols + 3)
    {
        if (row < rows && col < cols)
        {
            // convolving
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    float* inputValue = (float*)((char*)input + row*inputPitch) + col;

                    float* gxValue = (float*)((char*)gx + (row + i)*outputPitch) + (col + j);
                    // convolving gx
                    *gxValue += sobelKernelX[i][j]*(*inputValue);

                    float* gyValue = (float*)((char*)gy + (row + i)*outputPitch) + (col + j);
                    // convolving gy
                    *gyValue += sobelKernelY[i][j]*(*inputValue);
                }
            }          
        }

        float* gxValue = (float*)((char*)gx + row*outputPitch) + col;
        float* gyValue = (float*)((char*)gy + row*outputPitch) + col;
        float* outputValue = (float*)((char*)output + row*outputPitch) + col;
        *gxValue = *gxValue > 0 ? *gxValue : -*gxValue;
        *gyValue = *gyValue > 0 ? *gyValue : -*gyValue;
        //*outputValue = fabsf(*gxValue) + fabsf(*gyValue);
        *outputValue = *gxValue + *gyValue;
        //*outputValue = sqrtf(powf(*gxValue, 2) + powf(*gyValue, 2));
    }
}

void sobelOperator(const Mat & input, Mat & output)
{
    float* src; float* dst;

    cudaStream_t srcStream, dstStream, gxStream, gyStream;
    cudaStreamCreate(&srcStream); cudaStreamCreate(&dstStream); cudaStreamCreate(&gxStream); cudaStreamCreate(&gyStream);

    size_t srcPitch, dstPitch;
    cudaMallocPitch(&src, &srcPitch, sizeof(float)*input.cols, input.rows);
    cudaMemcpy2DAsync(src, srcPitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, srcStream);

    cudaMallocPitch(&dst, &dstPitch, sizeof(float)*output.cols, output.rows);
    cudaMemcpy2DAsync(dst, dstPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, dstStream);

    // store gx matrix and gy matrix
    float* gx; float* gy;
    size_t gxPitch, gyPitch;

    cudaMallocPitch(&gx, &gxPitch, sizeof(float)*output.cols, output.rows);
    cudaMemcpy2DAsync(gx, gxPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, gxStream);

    cudaMallocPitch(&gy, &gyPitch, sizeof(float)*output.cols, output.rows);
    cudaMemcpy2DAsync(gy, gyPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, gyStream);

    cudaStreamSynchronize(srcStream); cudaStreamSynchronize(dstStream); cudaStreamSynchronize(gxStream); cudaStreamSynchronize(gyStream);

    cudaMemset(gx, 0, sizeof(float)*output.rows*output.cols);
    cudaMemset(gy, 0, sizeof(float)*output.rows*output.cols);

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

    sobelOperator <<<blockSize, threadSize >>>(src, input.rows, input.cols, srcPitch, gx, gy, dst, dstPitch);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cout << cudaGetErrorString(error) << endl;
    }

    // get data back
    cudaMemcpy2D(output.data, sizeof(float)*output.cols, dst, dstPitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost);

    // resource releasing
    cudaFree(src); cudaFree(dst); cudaFree(gx); cudaFree(gy);
    cudaStreamDestroy(srcStream); cudaStreamDestroy(dstStream); cudaStreamDestroy(gxStream); cudaStreamDestroy(gyStream);
}

int main()
{
    string path = "type-c.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F);
    Mat result(Size(img.cols + 2, img.rows + 2), CV_32F, Scalar(0));

    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);
    sobelOperator(img, result);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    cout << "time cost on device: " << time << " ms." << endl;

    Mat sobel;
    double cpuStart = (double)getTickCount();
    Sobel(img, sobel, img.depth(), 1, 1);
    double cpuEnd = (double)getTickCount();
    double cpuTime = (cpuEnd - cpuStart) / getTickFrequency();
    cout << "time cost on cpu: " << cpuTime * 1000 << " ms." << endl;

    string title = "CUDA";
    result.convertTo(result, CV_8U);
    namedWindow(title);
    imshow(title, result);
    waitKey(0);
    imwrite("sobel.jpg", result);

    return 0;
}