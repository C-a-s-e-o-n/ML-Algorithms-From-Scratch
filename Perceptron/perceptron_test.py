import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from perceptron import Perceptron

if __name__ == "__main__":
    def accuracy(Y_true, Y_pred):
        accuracy = np.sum((Y_true == Y_pred) / len(Y_true))
        return accuracy

    X, Y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=123)

    p = Perceptron(learning_rate=0.01, n_iters = 1000)
    p.fit(X_train, Y_train)
    Y_pred = p.predict(X_test)

    acc = accuracy(Y_test, Y_pred)
    print(acc)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=Y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    ymin = np.amin(X_train[:,1])
    ymax = np.amax(X_train[:,1])
    ax.set_ylim([ymin-3, ymax+3])

    plt.show()
