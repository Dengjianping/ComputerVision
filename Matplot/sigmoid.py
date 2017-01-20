import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-10, 10, 0.5)
y = (1 - np.exp(-x)) / (1 + np.exp(-x))
plt.plot(x, y)
plt.show()