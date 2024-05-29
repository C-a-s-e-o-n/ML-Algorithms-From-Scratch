import numpy as np
from layer import Layer
from scipy import signal

# Note that the formula for calculating the output size of a convolution layer
# is as follows:
    # [(W - K + 2P) / S] + 1, where
    # W is the input volume, i.e. for MNIST, 28x28, W = 28
    # K is the Kernel Size, typically K = 3 or K = 5
    # P is the padding, which is typically zero for pure classification tasks,
    # S is stride, which is the speed of the kernels movement, typically 1

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # pytorch convention is to order input tuple as (color channel, height, width)
        input_depth, input_height, input_width = input_shape
        # number of filters in the conv layer, i.e., depth=32 means 32 filters
        # filters are analogous to weights, so 32 filters means 
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        # Ex: (32, 1, 3, 3) indicates 32 filters of size 3x3 applied to 1 channel
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.biases)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.depth):
                # scipy function to compute the cross-correlation (non-rotated convolution )
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                # dE / dK
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                # dE / dX
                input_gradient[i] = signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        # output_gradient = error = dE / db
        self.biases -= learning_rate * output_gradient
        return input_gradient
