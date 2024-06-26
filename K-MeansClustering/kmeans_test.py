import numpy as np 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from kmeans import KMeans


    X, Y= make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
    print(X.shape)

    clusters = len(np.unique(Y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=False)
    Y_pred = k.predict(X)

    k.plot()