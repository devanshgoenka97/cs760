import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()


plt.xlabel("X")
plt.ylabel("Y")

# Plotting L2 norm
plt.title("L-2 Norm")
# Plotting L-1 norm
#ax.add_patch(Rectangle((-1, 0), 1.414, 1.414, angle=45, color="red"))
# Plotting L-inf norm
#ax.add_patch(Rectangle((-1, -1), 2, 2, color="red"))
circle1 = plt.Circle((0, 0), 1, color='r')
ax.add_patch(circle1)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
ax.set_aspect('equal')

ax.grid(True, which='both')

plt.show()