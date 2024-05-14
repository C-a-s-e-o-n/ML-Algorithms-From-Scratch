import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from random_forest import RandomForest

def accuracy(Y_true, Y_pred):
    accuracy = np.sum(Y_true == Y_pred) / len(Y_true)
    return accuracy

data = datasets.load_breast_cancer()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

clf = RandomForest(n_trees=3)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
acc = accuracy(Y_test, Y_pred)

print("Accuracy: ", acc)