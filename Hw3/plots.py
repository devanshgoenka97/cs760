import matplotlib.pyplot as plt

x = [0,0,1/4,2/4,1]
y = [0, 1/3, 2/3, 1, 1]

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(0,1.2)
plt.ylim(0,1.2)
plt.plot(x, y, 'o-', color='red', markersize=8)
plt.show()
