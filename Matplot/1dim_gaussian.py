import numpy as np
import matplotlib.pyplot as plt

theta = 1.0
u = 2
x = np.arange(-5,5, 0.01)
y = np.sqrt(np.power(x,2) - 2)
a = np.sqrt(2 - np.power(x,2))
z = (1 / (np.sqrt(2*np.pi)*theta)) * np.exp(-np.power(x-u, 2) / (2*np.power(theta, 2)))
gaussian = plt.plot(x, z)
curve = plt.plot(x,y)
circle = plt.plot(x,a)
plt.show()