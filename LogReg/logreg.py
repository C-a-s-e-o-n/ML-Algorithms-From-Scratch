import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Predicting if someone would survive the Titanic Disaster with Logistic Regression

X_train = pd.read_csv("data/train_X.csv")
Y_train = pd.read_csv("data/train_Y.csv")

X_test = pd.read_csv("data/test_X.csv")
Y_test = pd.read_csv("data/test_Y.csv")

#print(X_train.head()) # Print the first few rows of the dataset

# Get rid of unneccessary column
X_train = X_train.drop("Id", axis=1)
Y_train = Y_train.drop("Id", axis=1)
X_test = X_test.drop("Id", axis=1)
Y_test = Y_test.drop("Id", axis=1)

# Returns numpy arrays of these pandas dataframes
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values

# Repshape to facillitate matrix multiplications
# X should be n x m, Y should be 1 x m
# n should be the number of features, m should be the number of rows
# Currently, the data is m x n, so we take the transpose
# Then, we take the second indice of the shape of x, (7, 891), which is 891, or m
# And we apply this to Y, which should be 1 x m

X_train = X_train.T
Y_train = Y_train.reshape(1, X_train.shape[1])
X_test = X_test.T
Y_test = Y_test.reshape(1, X_test.shape[1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def model(X_train, Y_train, alpha, iterations):
    n = X_train.shape[0]
    m = X_train.shape[1]

    # Initialize weights to a matrix of zeros which of n x 1 size 
    W = np.zeros((n, 1))
    # Initialize biases to a random scalar value
    b = 0

    cost_list = []

    for i in range(iterations):

        Z = np.dot(W.T, X_train) + b # Send this value to the sigmoid function to get the probability
        A = sigmoid(Z) # Returns a 1 x m matrix of probabilities for each training example using W and b

        J = -(1 / m) * (np.sum(Y_train * np.log(A)) + np.sum((1 - Y_train) * np.log(1 - A)))

        cost_list.append(J)

        if (i%(iterations/10) == 0):
            print("Cost after ", i, "iteration is : ", J)

        dJ_dW = (1/m) * np.dot(A - Y_train, X_train.T) # returns a matrix of shape 1 x n
        dJ_dB = (1/m)*np.sum(A - Y_train) # returns a matrix of shape 1 x m

        # Use to update variables simultaneously during gradient descent
        W = W - (alpha * dJ_dW.T)
        b = b - (alpha * dJ_dB)

    return W, b, cost_list

iterations = 100000
alpha = .0015 # Seems to be a good learning rate for this data based on the graph

W, b, cost_list = model(X_train, Y_train, alpha, iterations)

plt.plot(np.arange(iterations), cost_list)
plt.show()

def accuracy(X_test, Y_test, W, b):

    Z = np.dot(W.T, X_test) + b
    A = sigmoid(Z)

    A = A > 0.5 # Creates an array of boolean values

    A = np.array(A, dtype = 'int64') # Converts these values to integers

    # Sums up the absolute differences across all training examples of the
    # predicted probabilities (A) minus the actual values (Y_test) and
    # subtracts this by 1 and divides by the training examples to get the fraction
    # of correct values, * 100 for percentage
    accuracy = (1 - np.sum(np.absolute(A - Y_test)) / Y_test.shape[1]) * 100 

    print("Accuracy of the model is : ", accuracy, "%")

accuracy(X_test, Y_test, W, b)
