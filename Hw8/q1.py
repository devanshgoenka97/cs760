import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import seaborn as sns
from seaborn.miscplot import palplot

# Constructing threes from text file
three_vectors = np.zeros((200, 256))
counter = 0

with open('three.txt', 'r') as threes:
    for line in threes:
        vector = np.array([int(token) for token in line.split()])
        three_vectors[counter] = vector
        counter += 1

# Constructing eights from text file
eight_vectors = np.zeros((200, 256))
counter = 0

with open('eight.txt', 'r') as eights:
    for line in eights:
        vector = np.array([int(token) for token in line.split()])
        eight_vectors[counter] = vector
        counter += 1

# Visualizing two data points

#Image.fromarray(np.reshape(three_vectors[0], (16, 16))).show()
#Image.fromarray(np.reshape(eight_vectors[0], (16, 16))).show()

X = np.vstack((three_vectors, eight_vectors))
y = X.mean(axis = 0)

# Visualizing the sample mean
#Image.fromarray(np.reshape(y, (16, 16))).show()


# Calculating the sample covariance matrix
centered_X = X - y
sample_cov = np.cov(centered_X.T)
submatrix = sample_cov[0:5, 0:5]

# Performing eigendecomposition on the sample covariance
eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)

# Printing largest 2 eigenvalues
print(f"Largest Eigenvalue: {eigenvalues[-1]}")
print(f"Second largest Eigenvalue: {eigenvalues[-2]}")

# Getting 1st and 2nd Principal Components as the eigenvectors corresponding to the largest 2 eigenvalues
first_pc = np.reshape(eigenvectors[:, -1], (256, 1))
second_pc = np.reshape(eigenvectors[:, -2], (256, 1))

# Normalizing all values in eigenvectors to [0, 1] range
first_pc = (first_pc-min(first_pc))/(max(first_pc)-min(first_pc))
second_pc = (second_pc-min(second_pc))/(max(second_pc)-min(second_pc))

# Now expanding the range to [0, 255]
first_pc = first_pc * 255
second_pc = second_pc * 255

# Visualizing the two eigenvectors
#Image.fromarray(np.reshape(first_pc, (16, 16))).show()
#Image.fromarray(np.reshape(second_pc, (16, 16))).show()

V = np.hstack((np.reshape(eigenvectors[:, -1], (256, 1)), np.reshape(eigenvectors[:, -2], (256, 1))))

# Creating projection of X along the 2 principal components
proj = np.dot(centered_X, V)

first_three  = proj[0]
first_eight = proj[200]

print("First three on the 2 PCs:")
print(first_three)

print("First eight on the 2 PCs:")
print(first_eight)

# Constructing labels for plotting
labels = ['three'] * 200
labels.extend(['eight'] * 200)

sns.scatterplot(proj[:, 0], proj[:, 1], hue=labels, palette=['red', 'blue'])
plt.show()