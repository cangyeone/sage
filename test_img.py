import numpy as np 
import matplotlib.pyplot as plt

x1 = np.random.random([1000])
y1 = np.arange(1000)
x2 = np.random.random([1000])
y2 = np.arange(1000) + 2000 

x3 = np.zeros([3000])
x3[1000:2000] = 1 
x3[500:600] = 2 
y3 = np.arange(3000)

plt.plot(y1, x1+5)
plt.plot(y2, x2+5)
plt.plot(y3, x3)
plt.show()
