#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\host_vector.h"
#include "thrust\device_vector.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\cudaarithm.hpp>
#include <opencv2\cudalegacy.hpp>
#include <opencv2\cudaimgproc.hpp>

using namespace std;
using namespace cv;
//using namespace thrust;

__constant__ float PI = 3.1415;

/*float twoDimGaussian(int x, int y, float theta)
{
    float coeffient = 1 / (2 * PI*pow(theta, 2));
    float powerIndex = -(pow(x, 2) + pow(y, 2)) / (2 * pow(theta, 2));
    return coeffient*exp(powerIndex);
}*/

__device__ float twoDimGaussian(int x, int y, float theta)
{
    float coeffient = 1 / (2 * PI*powf(theta, 2));
    float powerIndex = -(powf(x, 2) + powf(y, 2)) / (2 * powf(theta, 2));
    return coeffient*expf(powerIndex);
}

/*void initGaussianMatrix(thrust::host_vector<thrust::host_vector<float> >* matrix, int radius, float theta = 1.0)
{
    for (size_t i = 0; i < 2 * radius + 1; i++)
    {
        thrust::host_vector<float> t;
        for (size_t j = 0; j < 2 * radius + 1; j++)
        {
            float gaussianValue = twoDimGaussian(j - radius, radius - i, theta);
            t.push_back(gaussianValue);
        }
        matrix->push_back(t);
    }
}*/

__global__ void gaussianBlur(uchar* input, int rows, int cols, uchar* output, int radius, float theta = 1.0)
{
    extern __shared__ float gaussian[];

    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < 2 * radius + 1 && col < 2 * radius + 1)
    {
        gaussian[row*(2 * radius + 1) + col] = twoDimGaussian(col - radius, radius - row, theta);
    }
    __syncthreads();

    if (row < rows&&col < cols)
    {
        for (size_t i = 0; i < 2 * radius + 1; i++)
        {
            for (size_t j = 0; j < 2 * radius + 1; j++)
            {
                output(row + i, col + j) += (float)input(row, col) * gaussian[i*(2 * radius + 1) + j];
            }
        }
    }
}

void gaussianBluring(const Mat & src, const Mat & des, int radius, float theta = 1.0)
{
    // define blocks size and threads size
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceCount - 1);
    int blockCount = (int)(des.rows * des.cols / prop.maxThreadsPerBlock) + 1;
    blockCount = (int)(sqrt(blockCount)) + 1;
    dim3 blockSize(blockCount, blockCount);
    dim3 threadSize(32, 32);

    // copy data to device
    int channelCount = src.channels();
    switch (channelCount)
    {
    case 1:
        uchar* srcData; float1* desData;
        cudaMalloc((void**)&srcData, sizeof(uchar)*src.rows*src.cols);
        cudaMalloc((void**)&desData, sizeof(uchar)*des.rows*des.cols);

        cudaStream_t srcStream, desStream;
        cudaStreamCreate(&srcStream); cudaStreamCreate(&desStream);
        cudaMemcpyAsync(srcData, src.data, sizeof(uchar)*src.rows*src.cols, cudaMemcpyHostToDevice, srcStream);
        cudaMemcpyAsync(desData, des.data, sizeof(float1)*des.rows*des.cols, cudaMemcpyHostToDevice, desStream);
        // block here until the data copy is finished
        cudaStreamSynchronize(srcStream); cudaStreamSynchronize(desStream);

        cudaMemset(desData, 0, sizeof(float1)*des.rows*des.cols);

        //call kernel function
        int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float);
        gaussianBlur<<<blockSize, threadSize, dynamicSize>>> ()

        // get data back to host
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
        {
            cout << cudaGetErrorString(error) << endl;
        }
        cudaMemcpy(des.data, desData, sizeof(float1)*des.rows*des.cols, cudaMemcpyDeviceToHost);

        // recource releasing
        cudaStreamDestroy(srcStream); cudaStreamDestroy(desStream);
        cudaFree(srcData); cudaFree(desData);
    default:
        break;
    }
}


int main(void)
{
    string path = "type-c.jpg";
    
    // source image
    _InputArray hostInput = imread(path, IMREAD_GRAYSCALE);
    cuda::GpuMat deviceInput = hostInput.getGpuMat();
    cout << path << endl;
    
    // gaussian kernel radius, the size is 2 * radius + 1, odd number is convenient for computing
    int radius = 2;
    
    InputArray hostResult = Mat(Size(deviceInput.cols + 2 * radius, deviceInput.rows + 2 * radius), CV_32F, Scalar(0));
    cuda::GpuMat deviceResult = hostResult.getGpuMat();

    // so the matrix size is 2 * radius + 1, use even number is convenient for computing.
    thrust::host_vector<thrust::host_vector<float> > hostGaussianMatrix;
    //initGaussianMatrix(&hostGaussianMatrix, radius);
    thrust::device_vector<thrust::device_vector<float> > deviceGaussianMatrix = hostGaussianMatrix;
    /*
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least, 
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */
    /*int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceCount - 1);
    //int blockCount = (int)(hostResult.rows * hostResult.cols / prop.maxThreadsPerBlock) + 1;*/
    //blockCount = (int)(sqrt(blockCount)) + 1;
    //dim3 blockSize(blockCount, blockCount);
    dim3 blockSize(17, 17);
    dim3 threadSize(32, 32);
    
    gaussianBlur <<<blockSize, threadSize>>> (deviceInput, deviceResult, deviceGaussianMatrix, radius);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cout << cudaGetErrorString(error) << endl;
    }

    //deviceResult.download(hostResult);
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