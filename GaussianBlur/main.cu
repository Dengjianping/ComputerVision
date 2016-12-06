#include <stdio.h>
#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

__constant__ float PI = 3.1415;

__device__ float twoDimGaussian(int x, int y, float theta)
{
    float coeffient = 1 / (2 * PI*powf(theta, 2));
    float powerIndex = -(powf(x, 2) + powf(y, 2)) / (2 * powf(theta, 2));
    return coeffient*expf(powerIndex);
}

__global__ void gaussianBlur(uchar* input, size_t srcPitch, int rows, int cols, float* output, size_t dstPitch, int radius, float theta = 1.0)
{
    // use a 1-dim array to store gaussian matrix
    extern __shared__ float gaussian[];

    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < rows&&col < cols)
    {
        // generate a gaussian matrix
        if (row < 2 * radius + 1 && col < 2 * radius + 1)
        {
            float sum = 0.0;
            gaussian[row*(2 * radius + 1) + col] = twoDimGaussian(col - radius, radius - row, theta);
            __syncthreads();           
        }

        for (size_t i = 0; i < 2 * radius + 1; i++)
        {
            for (size_t j = 0; j < 2 * radius + 1; j++)
            {
                // convolving, about how addressing matrix in device, 
                // see this link http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
                uchar *inputValue = (uchar *)((char *)input + row*srcPitch) + col;
                float *outputValue = (float *)((char *)output + (row + i)*dstPitch) + (col + j);                
                *outputValue += (float)(*inputValue) * gaussian[i*(2 * radius + 1) + j];
            }
        }
    }
}

void gaussianBlur(const Mat & src, const Mat & dst, int radius, float theta = 1.0)
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
    int blockCount = (int)(dst.rows * dst.cols / prop.maxThreadsPerBlock) + 1;
    blockCount = (int)(sqrt(blockCount)) + 1;
    dim3 blockSize(blockCount, blockCount);
    dim3 threadSize(32, 32);
    
    // create 2 streams to asynchronously copy data to device
    cudaStream_t srcStream, dstStream;
    cudaStreamCreate(&srcStream); cudaStreamCreate(&dstStream);

    // copy data to device
    uchar* srcData; float* dstData;
        
    size_t srcPitch;
    cudaMallocPitch(&srcData, &srcPitch, sizeof(uchar)*src.cols, src.rows);
    cudaMemcpy2DAsync(srcData, srcPitch, src.data, src.cols*sizeof(uchar), src.cols*sizeof(uchar), src.rows, cudaMemcpyHostToDevice, srcStream);
        
    size_t dstPitch;
    cudaMallocPitch(&dstData, &dstPitch, sizeof(float)*dst.cols, dst.rows);
    cudaMemcpy2DAsync(dstData, dstPitch, dst.data, dst.cols*sizeof(float), dst.cols*sizeof(float), dst.rows, cudaMemcpyHostToDevice, dstStream);
    cudaStreamSynchronize(srcStream); cudaStreamSynchronize(dstStream);
        
    int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float);
    gaussianBlur <<<blockSize, threadSize, dynamicSize>>> (srcData, srcPitch, src.rows, src.cols, dstData, dstPitch, radius, theta=1.5);
        
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        cout << cudaGetErrorString(error) << endl;
    }
    
    // copy data back to host
    cudaMemcpy2D(dst.data, sizeof(float)*dst.cols, dstData, dstPitch, sizeof(float)*dst.cols, dst.rows, cudaMemcpyDeviceToHost);
        
    // resource releasing
    cudaFree(srcData); cudaFree(dstData);

    cudaStreamDestroy(srcStream); cudaStreamDestroy(dstStream);
}


int main(void)
{
    string path = "type-c.jpg";
    
    // source image
    Mat hostInput = imread(path, IMREAD_GRAYSCALE);
    
    // gaussian kernel radius, the size is 2 * radius + 1, odd number is convenient for computing
    int radius = 2;
    Mat hostOutput(Size(hostInput.cols + 2 * radius, hostInput.rows + 2 * radius), CV_32F, Scalar(0));
    
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);

    cudaEventRecord(start);
    gaussianBlur(hostInput, hostOutput, radius);
    cudaEventRecord(end);
    cudaEventSynchronize(end); // hold here to wait all event finish

    float time;
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start); cudaEventDestroy(end);

    cout << "time cost on device: " << time << " ms" << endl;
    
    /* 
    need to convert to CV_8U type, because a CV_32F image, whose pixel value ranges from 0.0 to 1.0
    http://stackoverflow.com/questions/14539498/change-type-of-mat-object-from-cv-32f-to-cv-8u
    */
    Mat result;
    hostOutput.convertTo(result, CV_8U);

    Mat cpu;
    double cpuStart = (double)getTickCount();
    GaussianBlur(hostInput, cpu, Size(3, 3), 1.0);
    double cpuEnd = (double)getTickCount();
    double cpuTime = (cpuEnd - cpuStart) / getTickFrequency();
    cout << "time cost on cpu: " << cpuTime * 1000 << " ms" << endl; // millisecond return

    string title = "CUDA";
    namedWindow(title);
    imshow(title, result);

    waitKey(0);
    //system("pause");
    return 0;
}