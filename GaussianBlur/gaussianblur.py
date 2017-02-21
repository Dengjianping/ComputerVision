import cv2, math, time
import numpy as np

def twoDimGaussian(x, y, theta):
    return (1 / (2*math.pi*math.pow(theta, 2))) * (math.exp(-(x*x + y*y) / (2*theta*theta)))

def gaussianMatrix(radius, theta=1.0):
    # kernel = [[0 for i in range(2*radius+1)] for i in range(2*radius+1)]
    # kernel = np.array(kernel)
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype='float32')
    for i in range(size):
        for j in range(size):
            kernel[i][j] = twoDimGaussian(radius-i, j-radius, theta)
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
                    if (0 <= i+m-int(kernelRow/2) < inputRow and 0 <= j+n-int(kernelCol/2) < inputCol):
                        output[i+m-int(kernelRow/2)][j+n-int(kernelCol/2)] += input[i][j]*kernel[m][n]
    # output.dtype = 'uint8'           
    return output
    
if __name__ == '__main__':
    kernel = gaussianMatrix(2)

    path = r'type-c.jpg'
    title = r'cv'
    img = cv2.imread(path, 2)
    t1 = time.time()
    blur = convolve2D(img, kernel)
    print(time.time() - t1)
    
    cv2.namedWindow(title)
    cv2.imshow(title, img)
    cv2.imshow('blurred', blur)
    cv2.waitKey(0)