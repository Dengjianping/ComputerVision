import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

figure = plt.figure()
ax = Axes3D(figure)

x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
x, y = np.meshgrid(x, y)

z = np.exp(np.power(x,2) + np.power(y, 2))
ax.plot_surface(x, y, z, rstride=1,cstride=1, cmap='rainbow')
# plt.plot(x, y)
plt.show()