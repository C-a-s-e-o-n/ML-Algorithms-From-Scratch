import numpy as np
import random

def sigmoid(z):
    # z will be a vector wa + b representing the linearly transformed values for each neuron
    # note that b has been applied elementwise to each neuron
    # sigmoid is then applied elementwise to these elements to ensure everything is in the range [0,1]
    return 1.0 / (1 + np.exp(-z))

class Network(object):
    def __init__(self, sizes):
        # define networks as net = Network([3, 2, 5]), for each layer size
        self.num_layers = len(sizes)
        self.sizes = sizes

        # initializes random biases for layers after input layer
        # biases is a list of y x 1 column vectors
        # weights is a list of y x X vectors, where sizes[:-1] excludes the output layer
        # the zip function here pairs each layer together, and a weight vector is generated for each
        # ex: net = NetWork([3, 2, 5]) 
            # bias 1: random 2 x 1 vector for each neuron in the layer
            # bias 2: random 5 x 1 vector for each neuron in the layer
            # weight 1: 3 x 2 vector connecting input layer and hidden layer
            # weight 2: 2 x 5 vector connecting hidden layer and output layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feed_forward(self, a):
        # Returns the output of the network for input 'a', an n x 1 column vector
        # note that the zip function basically returns a zipped up iterator for tuples of linear combinations
        # of the two things you're combining, elementwise
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # training data is a list of tuples (x, y) representing data and its correct label
        # if test data is provided, then the network will be evaluated against the test data after each
        # epoch, and partial progress is printed; 
        # this is useful for tracking progress, but slows things down considerably
        if test_data: 
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))

