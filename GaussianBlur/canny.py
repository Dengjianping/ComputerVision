import cv2, math, time
import numpy as np
from numpy import linalg
from scipy import signal

def threshold(input, value):
    output = np.zeros(input.shape, dtype='uint8')
    rows, cols = input.shape
    for i in range(rows):
        for j in range(cols):
            if (input[i][j] >value):
                output[i][j] = 255
    return output

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

def gradient(input, orientation='x'):
    kernel = None
    if (orientation=='x'):
        kernel = np.array([[-1,1],[-1,1]])
    else:
        kernel = np.array([[-1, -1],[1,1]])
    output = convolve2D(input, kernel)
    return output

def amplitudeAndAngle(gradientX, gradientY):
    amplitude = np.sqrt(np.power(gradientX,2)+np.power(gradientY,2))
    angle = np.zeros(gradientY.shape, dtype='float32')
    rows, cols = gradientX.shape
    for i in range(rows):
        for j in range(cols):
            angle[i][j] = math.atan2(gradientY[i][j], gradientX[i][j])*180/math.pi
    return amplitude, angle

def canny(input, radius, theta):
    output = np.zeros(input.shape, dtype='uint8')
    blur = gaussianBlur(input, radius, theta)
    gradientX = gradient(blur, 'x')
    gradientY = gradient(blur, 'y')
    am, an = amplitudeAndAngle(gradientX, gradientY)
    return am, an

if __name__ == '__main__':
    path = r'type-c.jpg'
    title = r'cv'

    img = cv2.imread(path, 2)
    am, an = canny(img, 2,2)
    am.dtype = 'uint8'
    an.dtype = 'uint8'
    cv2.namedWindow(title)
    cv2.imshow(title, am)
    cv2.imshow('an', an)
    cv2.waitKey(0)