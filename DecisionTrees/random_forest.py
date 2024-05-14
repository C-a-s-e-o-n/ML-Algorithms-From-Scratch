import numpy as np
from collections import Counter

from decision_tree import DecisionTree

def bootstrap_sample(X, Y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True) # random int between 0 and n_samples, some replaced
    return X[idxs], Y[idxs]

def most_common_label(Y):
        counter = Counter(Y) # number of occurences of Y's, simialr to bincount
        most_common = counter.most_common(1)[0] # only the most common, as a tuple, [0] just gets value 
        return most_common

class RandomForest:
    def __init__(self, n_trees, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, Y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, Y_sample = bootstrap_sample(X, Y)
            tree.fit(X_sample, Y_sample)
            self.trees.append(tree)


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Now we swap axes, which changes the array shape (n_trees, n_samples) to 
        # (n_samples, n_trees) 
        # This allows for easier data aggregation, as each row represents a different tree, and columns represent predictions
        # Example for n_trees = 3, n_samples = 4
        # [1111 0000 1111]
        # [101 101 101 101]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # majority vote
        Y_pred = [most_common_label(tree_pred)[0] for tree_pred in tree_preds]
        return np.array(Y_pred)