import numpy as np
import matplotlib.pyplot as plt

# Linear Regression Model from Scratch using least squares error function / SLOW, BUT WORKS  #

# Sample House Training Data in 1D numpy arrays
x_train = np.array([1.0, 2.0]) # input variable, size of house in 1000s sq feet
y_train = np.array([300, 500]) # target, price in thousands


def calculate_model_output(x, y):
    m_y = y.shape # returns tuple of the dimension of y_train
    n = len(x)
    w = 0
    b = 0

    y_avg = np.mean(y)
    x_avg = np.mean(x)

    sum_of_xy = np.sum(x * y)
    x_sum_y_sum_over_n = (np.sum(y) * np.sum(x)) / n
    sum_xx = np.sum(x * x)
    x_sum_x_over_n = (np.sum(x) * np.sum(x)) / n

    w = (sum_of_xy - x_sum_y_sum_over_n) / (sum_xx - x_sum_x_over_n)
    b = y_avg - w * x_avg

    print (w, b)

    return w, b

w, b = calculate_model_output(x_train, y_train)

plt.plot(x_train, w * x_train + b, c='b', label='Prediction')

plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

plt.title('Housing Prices')
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sq feet)')
plt.legend()
plt.show()