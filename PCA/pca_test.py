from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from pca import PCA

data = datasets.load_iris()
X = data.data
Y = data.target

# X.shape = 150, 4
# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X: ', X.shape)
print('Shape of transformed X: ', X_projected.shape)

x1 = X_projected[:, 0] # first column
x2 = X_projected[:, 1] # second column

plt.scatter(x1, x2, 
            c=Y, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

plt.scatter(x1, x2)