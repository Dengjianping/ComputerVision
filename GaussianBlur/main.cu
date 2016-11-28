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

__device__ void gaussianKernel(cuda::GpuMat* gaussianMatrix, int row, int col, int radius, float theta)
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


int main(int argc, char** argv)
{
    string path = "1.jpg";

    Mat hostInput = imread(path, IMREAD_GRAYSCALE);
    cuda::GpuMat* deviceInput;
    deviceInput->upload(hostInput);
    Mat hostResult; cuda::GpuMat* deviceResult;

    // so the matrix size is 2 * radius + 1, use even number is convenient for computing.
    int radius = 2;
    dim3 blockSize(0, 0);
    dim3 threadSize(0, 0);
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