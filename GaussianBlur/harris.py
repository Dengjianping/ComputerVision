import cv2, math, time
import numpy as np
from numpy import linalg
from scipy import signal

def twoDimGaussian(x, y, theta):
    return (1 / (2 * math.pi * math.pow(theta, 2))) * (math.exp(-(x * x + y * y) / (2 * theta * theta)))

def gaussianMatrix(radius, theta=1.0):
    # kernel = [[0 for i in range(2*radius+1)] for i in range(2*radius+1)]
    # kernel = np.array(kernel)
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype='float32')
    for i in range(size):
        for j in range(size):
            kernel[i][j] = twoDimGaussian(radius - i, j - radius, theta)
    return kernel

def convolve2D(input, kernel):
    # input.dtype = 'float32'
    inputRow, inputCol = input.shape
    kernelRow, kernelCol = kernel.shape
    output = np.zeros(input.shape, dtype='uint8')

    for i in range(inputRow):
        for j in range(inputCol):
            for m in range(kernelRow):
                for n in range(kernelCol):
                    if (0 <= i + m - int(kernelRow / 2) < inputRow and 0 <= j + n - int(kernelCol / 2) < inputCol):
                        output[i + m - int(kernelRow / 2)][j + n - int(kernelCol / 2)] += input[i][j] * kernel[m][n]
    return output

def gaussianBlur(input, radius, theta):
    kernel = gaussianMatrix(radius, theta=theta)
    output = convolve2D(input, kernel)
    return output

def gradientX(input):
    kernel = np.array([[1, -1]])
    gradient = convolve2D(input, kernel)
    return gradient

def gradientY(input):
    kernel = np.array([[1], [-1]])
    gradient = convolve2D(input, kernel)
    return gradient

def blurGradientPower(gradient, radius, theta):
    gradient = np.power(gradient, 2)
    gradient = gaussianBlur(gradient, radius, theta)
    return gradient

def blurGradientXY(lx, ly, radius, theta):
    gradient = lx*ly
    gradient = gaussianBlur(gradient, radius, theta)
    return gradient

def maxR(image, gradientXX, gradientYY, gradientXY, k):
    row, col = image.shape
    max = 0
    for i in range(row):
        for j in range(col):
            m = np.array([[gradientXX[i][j], gradientXY[i][j]], [gradientXY[i][j], gradientYY[i][j]]])
            t = linalg.det(m) - k*(np.trace(m)**2)
            if (max < t):
                max = t

    for i in range(row):
        for j in range(col):
            m = np.array([[gradientXX[i][j], gradientXY[i][j]], [gradientXY[i][j], gradientYY[i][j]]])
            eigen = linalg.eig(m)[0]

if __name__ == '__main__':
    pass