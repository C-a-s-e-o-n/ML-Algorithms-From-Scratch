import numpy as np
import matplotlib.pyplot as plt

# Linear Regression Model from Scratch, NO ERROR FUNCTION #

# Sample House Training Data in 1D numpy arrays
x_train = np.array([1.0, 2.0]) # input variable, size of house in 1000s sq feet
y_train = np.array([300, 500]) # target, price in thousands
# m_num = len(x_train) # number of training examples

# Randomly set, untrained 
w = 100 # slope variable
b = 100 # intercept variable

def calculate_model_output(w, b, x_train):
    # Define a numpy array of zeros that is the same dimension as our input array
    m = x_train.shape # returns tuple of the dimension of x_train
    f_wb = np.zeros(m) 
    # Iterate through each training example, applying w and b to the y_hat forumla
    for i in range(len(x_train)):
        f_wb[i] = w * x_train[i] + b # fill numpy array with values from fnc

    return f_wb

test_f_wb = calculate_model_output(w, b, x_train)

plt.plot(x_train, test_f_wb, c='b', label='Prediction')

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

plt.title('Housing Prices')
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sq feet)')
plt.legend()
plt.show()