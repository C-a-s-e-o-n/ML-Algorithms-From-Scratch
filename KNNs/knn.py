import numpy as np
from collections import Counter

def eucliean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances    
        distances = [eucliean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k] # sort distances up to the k value from nearest to furthest
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1) 
        #most_common() returns the most common item in a list, and the amount as a tuple
        # we only want a number, so use slicing
        return most_common[0][0]