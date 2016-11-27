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

float twoDimGaussian(int row, int col, float theta = 1.0)
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
            cout << img.at<float>(i, j) << endl;
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
            cout << twoDimGaussian(j - radius, radius - i, theta) << endl;
            gaussianMatrix.at<float>(i, j) = twoDimGaussian(j - radius, radius - i, theta);
        }
    }

    // normalize gaussian matrix
    normalizeMatrix(gaussianMatrix);
    //normalize(gaussianMatrix, gaussianMatrix, 1.0, 1.0, NORM_L1);
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
            //cout << output.at<float>(i, j) << endl;
        }
    }
}

void convolutionMatrix1(Mat & input, Mat & kernel, Mat & output)
{
    for (size_t m = 0; m < kernel.rows; m++)
    {
        for (size_t n = 0; n < kernel.cols; n++)
        {
            for (size_t i = 0; i < input.rows; i++)
            {
                for (size_t j = 0; j < input.cols; j++)
                {
                    output.at<float>(i + m, j + n) += ((float)input.at<uchar>(i, j))*kernel.at<float>(m, n);
                }
            }
            //cout << output.at<float>(i, j) << endl;
        }
    }
}

Mat gaussianBlur(const Mat & input, const Mat & kernel)
{
    Mat output(Size(input.cols + kernel.cols - 1, input.rows + kernel.rows - 1), CV_8U, Scalar(0));

    for (size_t i = 0; i < output.rows; i++)
    {
        for (size_t j = 0; j < output.cols; j++)
        {
            float sum = 0.0;
            for (size_t m = 0; m <= i; m++)
            {
                for (size_t n = 0; n <= j; n++)
                {
                    if (m >= input.rows || n >= input.cols)continue;
                    if (i - m >= kernel.rows || j - n >= kernel.cols)continue;
                    if (i - m >= 0 && j - n >= 0)
                    {
                        sum += ((float)input.at<uchar>(m, n))*kernel.at<float>(i - m, j - n);
                    }
                }
            }
            if (sum > 255.0)
            {
                sum = 255.0;
            }
            output.at<uchar>(i, j) = (uchar)((int)sum);
        }
    }

    return output;
}

int main(int argc, char** argv)
{
    string path = "1.jpg";
    Mat input = imread(path, IMREAD_GRAYSCALE);
    vector<Mat> ch;
    split(input, ch);
    const int r = 3;
    Mat gaussianKenrel = gaussianKernel(r, r, 1.5);

    //Mat result(Size(input.cols + r - 1, input.rows + r - 1), CV_32F, Scalar(0));
    Mat result = gaussianBlur(input, gaussianKenrel);
    //convolutionMatrix1(input, gaussianKenrel, result);

    string title = "CUDA";
    namedWindow(title);
    imshow(title, result);

    waitKey(0);
    //system("pause");
    return 0;
}