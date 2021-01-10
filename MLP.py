import sys
import numpy as np
from sklearn import datasets
np.random.seed(0)

class MLP():

    def __init__(self, activations = ['ReLu', 'ReLu', 'Sigmoid'], num_input = 2, num_hidden= [2, 2], num_output = 1):

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

        for i in range(len(self.layers) - 1): #He initialisation to have better performance
            weight = np.random.normal(0, np.sqrt(2/self.layers[i]), (self.layers[i + 1], self.layers[i]))
            bias = np.random.normal(self.layers[i + 1],1)
            self.weights.append(weight)
            self.bias.append(bias)

    def sigmoid(self, x):
      return 1/(1 + np.exp(-x))

    def ReLu(self, x):
      return max(0, x)

    def feedforward(self, inputs):
        val = np.array(inputs)
        for j in range(len(self.layers) - 1):
            activated_op = []
            n_output = np.dot(val, self.weights[j].T) + self.bias[j]
            if self.activations[j] == "ReLu":
                for i in n_output:
                    activated_op.append(self.ReLu(i))
            else:
                for i in n_output:
                    activated_op.append(self.sigmoid(i))
            val = activated_op
        y_pred = val[0]
        return y_pred

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






