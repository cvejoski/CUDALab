import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.loadtxt(sys.argv[1])
T = np.loadtxt(sys.argv[2])

idx = np.argsort(X[:,0])
x = X[:,0][idx]
y = X[:,1][idx]
plt.scatter(T[:,0], T[:,1], color="y")
plt.plot(x, y, lw=2)
plt.show()
