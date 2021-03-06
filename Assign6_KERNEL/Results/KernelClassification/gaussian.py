import numpy as np
import matplotlib.pyplot as plt
import sys


X = np.loadtxt(sys.argv[1])
T = np.loadtxt(sys.argv[2])

##idx = np.argsort(X[:,0])
##x = X[:,0][idx]
##y = X[:,1][idx]
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('Kernel Classification (Gaussian Kernel) - Training Data (200 points), Test Data (1000 points), 3 Classes', fontsize=16)
use_colours = {"0": "red", "1": "green"}
plt.scatter(T[:,0], T[:,1], c=T[:,2], s=250, marker = "*",alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=X[:,2], alpha=0.5)
##plt.plot(x, y, lw=3, color="r")
plt.legend(('Training Data', 'Predicted Test Data'),'upper right', shadow=True)
plt.grid(True)
##plt.savefig('../../Plots/KernelRegression/destination_path.png', format='png')
plt.show()
