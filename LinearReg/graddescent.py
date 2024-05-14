import numpy as np
import matplotlib.pyplot as plt
import math

# Linear Regression Model from Scratch using least squares error function / SLOW, BUT WORKS  #

# Sample House Training Data in 1D numpy arrays
#x_train = np.array([1.0, 2.0, 3.0, 5.0, 7.0]) # input variable, size of house in 1000s sq feet
#y_train = np.array([300, 500, 700, 900, 1000]) # target, price in thousands

x_train = np.array([1.0, 2.0, 4.0, 4.3]) # input variable, size of house in 1000s sq feet
y_train = np.array([300, 500, 1000, 1100]) # target, price in thousands

#figure out how to graph everything like the stanford course does

w_init = 0 # Arbitrary starting values 
b_init = 1
alpha = .01 # Learning rate
MIN_THRESHOLD = .0001 # Define 10^-4 as the threshold for convergence
cost_init = 0
iterations = 0

def compute_gradient(x, y, w, b):
    dj_dw = 0 # Sum of partial derivatives w/ respect to w (slope)
    dj_db = 0 # Sum of partial derivatives w/ respect to b (intercept)

    m = x.size # Number of elements in np.array x_train

    for i in range(m):
        f_wb_i = w * x[i] + b # Linear Model for the ith data variable

        dj_db_i = f_wb_i - y[i] # ith partial derivative w/ respect to b
        dj_dw_i = (f_wb_i - y[i]) * x[i] # ith partial derivative w/ respect to w

        dj_db += dj_db_i # Add each derivative because the total derivative is a sum
        dj_dw += dj_dw_i

    dj_db = dj_db / m # Divide the derivatives by the # of training examples
    dj_dw = dj_dw / m

    return dj_dw, dj_db

def compute_cost(x, y, w, b, cost):
    m = x.size

    cost_i = 0

    for i in range(m):
        f_wb_i = w * x[i] + b

        cost_i = (f_wb_i - y[i]) ** 2

        cost += cost_i

    cost = cost * (1 / 2 * m)

    return cost

def gradient_descent(alpha, x, y, w, b, dj_dw, dj_db):
    norm = 0 # Calculate magnitude of the gradient using the Eucliean Norm

    while(True):
        w_tmp = w - (alpha * dj_dw)
        b_tmp = b - (alpha * dj_db)

        w = w_tmp
        b = b_tmp

        dj_dw, dj_db = compute_gradient(x, y, w, b) # Using new parameters, calculate new gradients

        norm = math.sqrt(dj_dw ** 2)

        if norm < MIN_THRESHOLD: # check if Eucliean Norm exceeds the minimum threshold for convergence
            break;
    
    return w, b

dj_dw, dj_db = compute_gradient(x_train, y_train, w_init, b_init)

w, b = gradient_descent(alpha, x_train, y_train, w_init, b_init, dj_dw, dj_db)

cost = compute_cost(x_train, y_train, w, b, cost_init)

plt.plot(x_train, w * x_train + b, c='b', label='Prediction')

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

plt.title('Housing Prices')
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sq feet)')
plt.legend()
plt.show()