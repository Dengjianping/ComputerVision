#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\cudaarithm.hpp>
#include <opencv2\cudalegacy.hpp>
#include <opencv2\cudaimgproc.hpp>

using namespace std;
using namespace cv;

__constant__ float PI = 3.1415;

__device__ float twoDimGaussian(int x, int y, float theta)
{
    float coeffient = 1 / (2 * PI*powf(theta, 2));
    float powerIndex = -(powf(x, 2) + powf(y, 2)) / (2 * powf(theta, 2));
    return coeffient*expf(powerIndex);
}

/*__device__ void gaussianKernel(cuda::GpuMat* gaussianMatrix, int row, int col, int radius, float theta)
{
    // Mat gaussianMatrix(Size(2 * radius + 1, 2 * radius + 1), CV_32F, Scalar(0));
    if (row < gaussianMatrix->rows&&col < gaussianMatrix->cols)
    {
        gaussianMatrix->ptr<float>(row)[col] = twoDimGaussian(col - radius, radius - row, theta);

        float sum = 0.0;
        if (row < gaussianMatrix->rows&&col < gaussianMatrix->cols)
        {
            sum += gaussianMatrix->ptr<float>(row)[col];
            gaussianMatrix->ptr<float>(row)[col] = gaussianMatrix->ptr<float>(row)[col] / sum;
        }
    }
}

__global__ void gaussianBlur(cuda::GpuMat* input, cuda::GpuMat* output, int radius, float theta = 1.0)
{
int row = blockDim.y*blockIdx.y + threadIdx.y;
int col = blockDim.x*blockIdx.x + threadIdx.x;

if (row < input->rows&&col < input->cols)
{
cuda::GpuMat gaussianMatrix(2 * radius + 1, 2 * radius + 1, CV_32F, Scalar(0));
gaussianKernel(&gaussianMatrix, row, col, radius, theta);
for (size_t i = 0; i < gaussianMatrix.rows; i++)
{
for (size_t j = 0; j < gaussianMatrix.cols; j++)
{
output->ptr<float>(row + i)[col + j] += ((float)input->ptr<uchar>(row)[col]) * gaussianMatrix.ptr<float>(i)[j];
}
}
}
}

*/

__device__ void gaussianKernel(cuda::PtrStepSzf gaussianMatrix, int row, int col, int radius, float theta)
{
    // Mat gaussianMatrix(Size(2 * radius + 1, 2 * radius + 1), CV_32F, Scalar(0));
    if (row < gaussianMatrix.rows&&col < gaussianMatrix.cols)
    {
        gaussianMatrix(row, col) = twoDimGaussian(col - radius, radius - row, theta);

        float sum = 0.0;
        if (row < gaussianMatrix.rows&&col < gaussianMatrix.cols)
        {
            sum += gaussianMatrix(row, col);
            gaussianMatrix(row, col) = gaussianMatrix(row, col) / sum;
        }
    }
}

__global__ void gaussianBlur(cuda::PtrStepSzf input, cuda::PtrStepSzf output, cuda::PtrStepSzf kernel, int radius, float theta = 1.0)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < input.rows&&col < input.cols)
    {
        //cuda::PtrStepSzf gaussianMatrix(2 * radius + 1, 2 * radius + 1, CV_32F, Scalar(0));
        gaussianKernel(kernel, row, col, radius, theta);
        for (size_t i = 0; i < kernel.rows; i++)
        {
            for (size_t j = 0; j < kernel.cols; j++)
            {
                output(row + i, col + j) += (float)input(row, col) * kernel(i, j);
            }
        }
    }
}


int main(int argc, char** argv)
{
    string path = "type-c.jpg";

    InputArray hostInput = imread(path, IMREAD_GRAYSCALE);
    cuda::GpuMat deviceInput;
    deviceInput = hostInput.getGpuMat();
    int radius = 2;
    InputArray hostResult(Size(deviceInput.cols + 2 * radius, deviceInput.rows + 2 * radius), CV_32F, Scalar(0));
    cuda::GpuMat* deviceResult;

    // so the matrix size is 2 * radius + 1, use even number is convenient for computing.
    
    /*
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least, 
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceCount - 1);
    int blockCount = ceil(hostResult.rows * hostResult.cols / prop.maxThreadsPerBlock);
    blockCount = ceil(sqrt(blockCount));
    dim3 blockSize(blockCount, blockCount);
    dim3 threadSize(32, 32);
    
    //gaussianBlur <<<blockSize, threadSize>>> (deviceInput, deviceResult, radius);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cout << cudaGetErrorString(error) << endl;
    }

    deviceResult->download(hostResult);
    //Mat result(Size(input.cols + 2 * radius, input.rows + 2 * radius), CV_32F, Scalar(0));
    //convolutionMatrix(input, gaussianKenrel, result);
    
    /* 
    need to convert to CV_8U type, because a CV_32F image, whose pixel value ranges from 0.0 to 1.0
    http://stackoverflow.com/questions/14539498/change-type-of-mat-object-from-cv-32f-to-cv-8u
    */
    //result.convertTo(result, CV_8U);

    string title = "CUDA";
    namedWindow(title);
    imshow(title, hostInput);

    waitKey(0);
    //system("pause");
    return 0;
}