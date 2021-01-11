import sys
import numpy as np
from sklearn import datasets
np.random.seed(0)

class MLP():

    def __init__(self, activations = ['Sigmoid', 'Sigmoid', 'Sigmoid'], num_input = 2, num_hidden= [2, 2], num_output = 1):

        #Initialise neural network

        self.activations = activations
        self.num_input = 2
        self.num_hidden = num_hidden
        self.num_output = num_output

        #Layers of the neural network
        self.layers = [num_input] + num_hidden + [num_output]

        #Initialise the weights
        self.weights = []
        self.bias = []

        #store the activations
        activations_vals = []
        for i in range(len(self.layers)):
            a = np.zeros(self.layers[i])
            activations_vals.append(a)
        self.activations_vals = activations_vals

        #store the gradients
        derivatives = []
        for i in range(len(self.layers) - 1):
            d = np.zeros((self.layers[i], self.layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        weights = []
        for i in range(len(self.layers) - 1):
            w = np.random.normal(0, np.sqrt(2/self.layers[i + 1]), (self.layers[i], self.layers[i + 1]))
            weights.append(w)
        self.weights = weights
        self.bias = np.random.rand(self.layers[i], 1)

    def sigmoid(self, x):
      return 1/(1 + np.exp(-x))

    def ReLu(self, x):
        for i in range(len(x)):
            x[i] = max(0, x[i])
            print(x[i])
        return x

    def LeakyReLu(self, x):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0.01*x
        return x

    def feedforward(self, inputs):
        act = inputs

        # save the activations for backpropogation
        self.activations_vals[0] = act

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(act, w)

            # apply sigmoid activation function
            act = self.sigmoid(net_inputs) if self.activations[i] == 'Sigmoid' else self.ReLu(net_inputs)

            # save the activations for backpropogation
            self.activations_vals[i + 1] = act

        # return output layer activation
        return act[0]

    def backpropagate(self):
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)
if __name__ == "__main__":
    # Data to feed the Neural Network
    data = datasets.make_circles(n_samples=100, shuffle=True, noise=0.10, random_state=None,
                                 factor=0.25)  # Generate annulus data

    x1 = data[0][:, 0]  # Features
    x2 = data[0][:, 1]

    inputs = []  # Zip the inputs together i.e [x1, x2]
    for x1, x2 in zip(x1, x2):
        inputs.append([x1, x2])

    y = data[1]  # Ground truth

    mlp = MLP()
    output = mlp.feedforward(inputs[0])

    print("Network activation: {}".format(output))







