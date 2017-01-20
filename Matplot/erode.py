import numpy as np

a=[0,0,1,1,1,0,0]
b=[0,0,1]
g=[1,0,0]

c = np.convolve(a, b, 'same')
print(a)

d = np.logical_and(a,c) * 1
print(d)