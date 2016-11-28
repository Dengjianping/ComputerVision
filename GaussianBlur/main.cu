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

const float PI = 3.1415;

__device__ float twoDimGaussian(int row, int col, float theta = 1.0)
{
    float coeffient = 1 / (2 * PI*pow(theta, 2));
    float powerIndex = -(pow(row, 2) + pow(col, 2)) / (2 * pow(theta, 2));
    return coeffient*exp(powerIndex);
}

void normalizeMatrix(Mat & img)
{
    float sum = 0.0;
    for (size_t i = 0; i < img.rows; i++)
    {
        for (size_t j = 0; j < img.cols; j++)
        {
            sum += (float)img.at<float>(i, j);
        }
    }
    cout << "sum: " << sum << endl;
    for (size_t i = 0; i < img.rows; i++)
    {
        for (size_t j = 0; j < img.cols; j++)
        {
            img.at<float>(i, j) = img.at<float>(i, j) / sum;
        }
    }
}

Mat gaussianKernel(int rows, int cols, float theta = 1.0)
{
    Mat gaussianMatrix(Size(cols, rows), CV_32F, Scalar(0));
    int radius = rows / 2;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            gaussianMatrix.at<float>(i, j) = twoDimGaussian(j - radius, radius - i, theta);
        }
    }

    // normalize gaussian matrix
    normalizeMatrix(gaussianMatrix);
    return gaussianMatrix;
}

void convolutionMatrix(Mat & input, Mat & kernel, Mat & output)
{
    for (size_t i = 0; i < input.rows; i++)
    {
        for (size_t j = 0; j < input.cols; j++)
        {
            for (size_t m = 0; m < kernel.rows; m++)
            {
                for (size_t n = 0; n < kernel.cols; n++)
                {
                    output.at<float>(i + m, j + n) += ((float)input.at<uchar>(i, j))*kernel.at<float>(m, n);                 
                }
            }
        }
    }
}

__global__ void gaussianBlur(cuda::GpuMat* input, cuda::GpuMat* kernel, cuda::GpuMat* output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    
    for (size_t i = 0; i < kernel.rows; i++)
    {
        for (size_t j = 0; j < kernel.cols; j++)
        {
            output.at<float>(row + i, col + j) += ((float)input.at<uchar>(row, col)) * kernel.at<float>(i, j);
        }
    }
}


int main(int argc, char** argv)
{
    string path = "1.jpg";
    Mat input = imread(path, IMREAD_GRAYSCALE);
    vector<Mat> ch;
    split(input, ch);
    const int r = 3;
    Mat gaussianKenrel = gaussianKernel(r, r, 1.5);

    Mat result(Size(input.cols + r - 1, input.rows + r - 1), CV_32F, Scalar(0));
    convolutionMatrix(input, gaussianKenrel, result);
    
    /* 
    need to convert to CV_8U type, because a CV_32F image, whose pixel value ranges from 0.0 to 1.0
    http://stackoverflow.com/questions/14539498/change-type-of-mat-object-from-cv-32f-to-cv-8u
    */
    result.convertTo(result, CV_8U);

    string title = "CUDA";
    namedWindow(title);
    imshow(title, result);

    waitKey(0);
    //system("pause");
    return 0;
}