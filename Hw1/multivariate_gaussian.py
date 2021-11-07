import matplotlib.pyplot as plt
import numpy as np

fig, ax  = plt.subplots()
mean = [1, -1]
cov = [[2, 0], [0, 2]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
ax.set_xlim([-8, 8.00])
ax.set_ylim([-8, 8.00])
ax.set_xbound(-8, 8)
ax.set_ybound(-8, 8)
ax.plot(x, y, 'bo', markersize = 10)
ax.grid(True, which='both')
plt.show()