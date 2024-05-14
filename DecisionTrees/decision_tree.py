import numpy as np
from collections import Counter

# Entropy is a measure of impurity/disorder in a set of labels
def entropy(Y):
    hist = np.bincount(Y) # counts # of occurrences of all class labels
    ps = hist / len(Y) # p(Y) for entropy calculation, list of probabilities (%) of occurrence for each class label
    return -np.sum([p * np.log2(p) for p in ps if p > 0]) # log undefined for negatives

class Node:
    # The asterisk indicates that value is a keyword arg, which means it can only be passed if you specify its name
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self): # leaf nodes are nodes which have values / have no branching nodes
        return self.value is not None
    
class DecisionTree:
    # n_feats is for only looping over a subset of features instead of all of them (greedy search)
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, Y):
        # grow tree
        # X.shape[1] means 2nd dimension of np array, # of features
        # n_feats can only be = to X.shape[1] or lower (its a subset of features)
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) 
        self.root = self._grow_tree(X, Y) # helper function

    def _grow_tree(self, X, Y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(Y)) # different class labels

        # base criteria for recursion
        # chooses the node we are satisfied with stopping at for a particular class label
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(Y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False) # random val between first 2 params, no repeating

        # greedy search
        best_feat, best_thresh = self._best_criteria(X, Y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], Y[left_idxs], depth+1) # recursive search of each side to find best 
        right = self._grow_tree(X[right_idxs, :], Y[right_idxs], depth+1) # features, threshold, and indxs
        return Node(best_feat, best_thresh, left, right) # construct tree, where left/right = child pointers

    # searches every possible (unique) criteria and chooses the best information gain
    def _best_criteria(self, X, Y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # only unique criteria 
            for threshold in thresholds:
                gain = self._information_gain(Y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        
        return split_idx, split_thresh
    
    # IG = E(parent) - [weighted average] * E(children)
    # We want a high information gain, which means we want low child entropy avg
    def _information_gain(self, Y, X_column, split_thresh):
        # parent E
        parent_entropy = entropy(Y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # weighted average of child E's
        n = len(Y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(Y[left_idxs]), entropy(Y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # return ig
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() # returns a 1-d list of items where this = true
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        # traverse tree
        return np.array([self._traverse_tree(x, self.root)[0] for x in X]) # start at the top of the tree
    
    def _traverse_tree(self, x, node):
        # base case
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _most_common_label(self, Y):
        counter = Counter(Y) # number of occurences of Y's, simialr to bincount
        most_common = counter.most_common(1)[0] # only the most common, as a tuple, [0] just gets value 
        return most_common

