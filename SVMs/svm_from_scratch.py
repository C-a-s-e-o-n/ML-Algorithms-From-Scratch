import numpy as np

# LINEAR BINARY CLASSIFICATION SVM (Support Vector Machine) FROM SCRATCH

# Equation for hyperplane: w * x - b = 0
# Equation for upper margin: w * x - b = 1
# Equation for lower margin: w * x - b = -1

# Constraints:
    # w * x[i] - b >= 1 if y[i] == 1
    # w * x[i] - b <= 1 if y[i] == -1

# Combining this, since -1 * -1 == +1:
    # y[i] * (w * x[i] - b) >= 1 for all y[i]

# Goal: to minimize our hinge loss and the regularization term whilst satisfying our constraint
# Hinge Loss: max(0, 1 - y[i] * y_hat[i]), where y_hat is predicted, and y[i] is true
    # This function returns 0 for correct classifications, which is why we want to minimize

# The regularization term is lambda times the norm of w squared, to penalize large weights

# The overall cost function is J = HingeLoss + RegularizationTerm; this means that for a correct classification,
# the only thing we are trying to minimize becomes the regularization term. This allows our SVM to strike a balance 
# between prioritizing maximizing the margin, and simplifying the overall weights. This is crucial.

class SVM:
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate # step size for our gradient descent
        self.lambda_param = lambda_param # regularization term with squared norm of w
        self.n_iters = n_iters # when to declare convergence for grad descent
        self.w = None 
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1) # where y <= 0, convert number to -1, else, convert to 1
        _, n_features = X.shape # returns m x n, where n is what we care about, bcuz we want w to have same # of columns

        # w is a column vector n x 1, and x[i] is a row vector 1 x k of data points for a specific feature 
        self.w = np.zeros(n_features) # create an array the same size as x for multiplication to be possible
        self.b = 0

        # GRADIENT DESCENT 
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # checks to see if the constraint holds, indicating a correct classification
                # if the constraint holds, we only subtract the reg term, a small amount
                # if it doesn't hold, we subtract by hinge loss AND reg term, a large amount, to penalize 
                # the equations in the conditionals represent the gradient of the reg term, and the gradient of the hinge loss 
                constraint = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 # basically the same as max(0, 1-y{i}*y_hat)
                if constraint:
                    self.w -= self.lr * (2 * self.lambda_param * self.w) # only the regularization term
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx] # note that the actual position of the hyperplane is only updated here 

    def predict(self, X): 
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output) # positive sign denotes positive class, else negative class
