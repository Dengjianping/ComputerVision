import matplotlib.pyplot as plt
import numpy as np
import math

x=1
y=2

a = np.array([3,5])
c = np.array([x,y])

x_range = np.linspace(2*-np.pi, 2*np.pi, 256, endpoint=True)
for i in range(10):
    c = c + i
    a = a + c
    hough = c[0]*np.cos(x_range) + c[1]*np.sin(x_range)
    hough1 = a[0]*np.cos(x_range) + a[1]*np.sin(x_range)
    plt.plot(x_range, hough)
    plt.plot(x_range, hough1)

p = math.sqrt(x**2 + y**2)
theta = math.atan2(y,x)
# theta = a/math.pi * 180.0

plt.show()