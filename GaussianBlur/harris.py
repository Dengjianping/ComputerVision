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
    # kernel = np.array([[-1, 0, 1],[-1,0,1],[-1,0,1]])
    gradient = convolve2D(input, kernel)
    return gradient

def gradientY(input):
    kernel = np.array([[1], [-1]])
    # kernel = np.array([[-1,-1,-1],[0,0,0], [1,1,1]])
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

def r(gradientXX, gradientYY, gradientXY, i, j, k):
    a = [-2,-1,0,1, 2]
    window = np.zeros((5,5), dtype='float32')
    for m in a:
        for n in a:
            if (m!=0 and n!=0):
                z = np.array([[gradientXX[i+m][j+n], gradientXY[i+m][j+n]], [gradientXY[i+m][j+n], gradientYY[i+m][j+n]]])
                t = linalg.det(z) - k * (np.trace(z) ** 2)
                window[m][n] = t
    return np.max(window)

def maxRv(image, gradientXX, gradientYY, gradientXY, k, repression, window):
    row, col = image.shape
    maxR = 0
    for i in range(row):
        for j in range(col):
            m = np.array([[gradientXX[i][j], gradientXY[i][j]], [gradientXY[i][j], gradientYY[i][j]]])
            t = linalg.det(m) - k*(np.trace(m)**2)
            if (maxR < t):
                maxR = t
    point = []
    for i in range(row):
        for j in range(col):
            if (window-1 < i < row-int(window/2) and window-1 < j < col -int(window/2)):
                m = np.array([[gradientXX[i][j], gradientXY[i][j]], [gradientXY[i][j], gradientYY[i][j]]])
                # eigen = linalg.eig(m)[0]
                t = linalg.det(m) - k * (np.trace(m) ** 2)
                if (t > repression*maxR and t > r(gradientXX, gradientYY, gradientXY, i,j,k)):
                    cor = (i, j)
                    point.append(cor)
    return point

if __name__ == '__main__':
    radius = 3
    theta = 2.0

    k = 0.06
    repression = 0.01
    window = 5

    path = r'image.jpg'
    title = r'cv'

    img = cv2.imread(path, 2)
    #img = None
    #cv2.cvtColor(image, img, cv2.COLOR_BGR2GRAY)
    x = gradientX(img)
    print(1)
    y = gradientY(img)
    print(2)
    xx = blurGradientPower(x,radius, theta)
    print(3)
    yy = blurGradientPower(y,radius, theta)
    print(4)
    xy = blurGradientXY(x,y, radius, theta)
    print(5)
    point = maxRv(img, xx,yy,xy, k, repression, window)
    a = [ -1, 0, 1]
    for p in point:
        print(p)
        i, j = p
        for p in a :
            for q in a:
                img[i+p][j+q] = 188

    title = 'harris'
    cv2.namedWindow(title)
    cv2.imshow(title, img)
    cv2.waitKey(0)