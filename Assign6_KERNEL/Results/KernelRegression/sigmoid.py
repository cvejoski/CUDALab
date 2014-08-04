import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.loadtxt(sys.argv[1])
T = np.loadtxt(sys.argv[2])

idx = np.argsort(X[:,0])
x = X[:,0][idx]
y = X[:,1][idx]
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kernel Regression (Sigmoid Kernel) - Training Data (500 points), Test Data (2000 points)', fontsize=16)
plt.scatter(T[:,0], T[:,1], color="b", alpha=0.5)
plt.plot(x, y, lw=3, color="r")
plt.legend(('Kernel Regr. curve over the test data', 'Training Data'),'upper right', shadow=True)
plt.grid(True)
##plt.savefig('../../Plots/KernelRegression/destination_path.png', format='png')
plt.show()
