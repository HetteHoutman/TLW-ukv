import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

random = np.random.uniform(-1.5, 1.5, (50,2))
def plus_sin(xy):
    return np.sin(xy[:,0] - xy[:,1])

z = plus_sin(random)

X = np.linspace(-1, 1, 10)
Y = X.copy()

XY = np.moveaxis(np.array(np.meshgrid(X,Y)), 0, 2)
Z = griddata(random, z, XY)
plt.contourf(XY[:,:,0], XY[:,:,1], Z)
plt.show()
