import numpy as np
import matplotlib.pyplot as plt
import random

mean1 = [5, 0]
cov1 = [[1, 0.25], [0.25, 1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T

mean2 = [-5, 0]
cov2 = [[1, -0.25], [-0.25, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T

x, y = [], []
for i in range(0, 100):
    t = random.uniform(0, 100)
    if random.uniform(0, 1) <= float(0.3):
        x.append(x1[t])
        y.append(y1[t])
    else:
        x.append(x2[t])
        y.append(y2[t])

fig, ax = plt.subplots()
ax.plot(x, y, 'bo', markersize = 10)
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])
plt.show()