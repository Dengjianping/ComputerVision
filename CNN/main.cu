#include <stdio.h>
#include <iostream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "curand.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>

#define MAX_THREADS 32

using namespace std;
using namespace cv;

void randCUDA()
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

}

void unnanedFunction(const Mat & img)
{
    int channelCount = img.channels();
    vector<Mat> chs;
    split(img, chs);

    float* channelData1; float* channelData2; float* channelData3;
    cudaStream_t channelStream1; cudaStream_t channelStream2; cudaStream_t channelStream3;
    size_t channelPitch1; size_t channelPitch2; size_t channelPitch3;

    // allocate channel 1
    cudaMallocPitch(&channelData1, &channelPitch1, sizeof(float)*img.cols, img.row);
    cudaMemcpy2DAsync(channelData1, channelPitch1, chs[0].data, sizeof(float)*img.cols, sizeof(float)*img.rows, img.rows, cudaMemcpyHostToDevice, channelStream1);

    cudaMallocPitch(&channelData2, &channelPitch2, sizeof(float)*img.cols, img.row);
    cudaMemcpy2DAsync(channelData2, channelPitch2, chs[1].data, sizeof(float)*img.cols, sizeof(float)*img.rows, img.rows, cudaMemcpyHostToDevice, channelStream2);

    cudaMallocPitch(&channelData3, &channelPitch3, sizeof(float)*img.cols, img.row);
    cudaMemcpy2DAsync(channelData3, channelPitch3, chs[2].data, sizeof(float)*img.cols, sizeof(float)*img.rows, img.rows, cudaMemcpyHostToDevice, channelStream3);


}

int main()
{
    string path = "type-c.jpg";
    Mat img = imread(path);

    string title = "CUDA";
    namedWindow(title);
    imshow(title, img);
    waitKey(0);

    return 0;
}