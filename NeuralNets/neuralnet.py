import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        # define networks as net = Network([3, 2, 5]), for each layer size
        self.num_layers = len(sizes)
        self.sizes = sizes

        # initializes random biases for layers after input layer
        # biases is a list of y x 1 column vectors
        # weights is a list of y x X matrices, where sizes[:-1] excludes the output layer
        # the zip function here pairs each layer together, and a weight vector is generated for each
        # ex: net = NetWork([3, 2, 5]) 
            # bias 1: random 2 x 1 vector for each neuron in the layer
            # bias 2: random 5 x 1 vector for each neuron in the layer
            # weight 1: 3 x 2 vector connecting input layer and hidden layer
            # weight 2: 2 x 5 vector connecting hidden layer and output layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        # Returns the output of the network for input 'a', an n x 1 column vector
        # note that the zip function basically returns a zipped up iterator for tuples of linear combinations
        # of the two things you're combining, elementwise
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

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
            # creates randomized subsets of length mini_batch_size, with that step size
            # example: n=1000, mini_batch_size = 50: [0 to 50], [50 to 100], etc
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))

    def update_mini_batch(self, mini_batch, eta):
        # update the network's weights and biases by applying gradient descent using backprop
        # to a single mini batch
        # mini_batch is a list of tuples (x, y) and eta is the learning rate

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnd for nb, dnd in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    #### Miscellaneous functions
def sigmoid(z):
    # z will be a vector wa + b representing the linearly transformed values for each neuron
    # note that b has been applied elementwise to each neuron
    # sigmoid is then applied elementwise to these elements to ensure everything is in the range [0,1]
    return 1.0 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))