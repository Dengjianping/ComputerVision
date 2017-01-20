import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

figure = plt.figure()
ax = Axes3D(figure)

theta = 2.5
x = np.arange(-10, 10, 0.25)
y = np.arange(-10, 10, 0.25)
x, y = np.meshgrid(x, y)

z = (1 / (np.pi*np.power(theta, 4)))*(1 - (np.power(x, 2)+np.power(y,2))/(2*np.power(theta,2)))*np.exp(-(np.power(x,2)+np.power(y,2))/(2*np.power(theta,2)))
z = np.sin(z)
ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='rainbow')

plt.show()