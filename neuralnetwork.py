import numpy as np
from random import shuffle

np.random.seed(0)

class NeuralNetwork:

    def __init__(self, layers, neurons, activations, inputs_x, inputs_y):

        #Checks
        assert(layers >= 1, 'Neural Network needs at least one layer')
        assert(layers == len(neurons), 'Please specify no. of neurons for all layers')
        assert(activations != '', 'Please specify activation function')
        assert(len(activations) != layers, 'Please specify activation function')
        assert (len(inputs_x) >= 1, 'Input needs more than 1 value')
        assert(len(inputs_x) == len(inputs_y), 'x_train and y_train are of different size')

        #Initialisation
        self.layers = layers
        self.neurons = [inputs_x.shape[1]] + neurons
        self.activations = activations
        self.inputs_x = inputs_x
        self.inputs_y = inputs_y
        self.weights = []
        self.bias = []

        #Initialise the weights and biases
        for i in range(len(neurons)):
            self.weights.append(np.random.normal(0, np.sqrt(2/neurons[i]), (neurons[i+1], neurons[i]))) #He initialisation
            self.bias.append(np.random.normal(neurons[i+1], 1)) #Bias for the layers

    def Relu(self, x): #ReLu activation function
        return max(0, x)

    def derv_ReLu(self, x): #Derivative of ReLu
        if x < 0:
            return 0
        elif x > 0:
            return x

    def sigmoid(self, x): #Sigmoid activation function
        return 1/(1 + np.exp(-x))

    def derv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def dldy(self, y_pred, y):
        return (y_pred - y)/y_pred(1 - y_pred)

    def dada(self, x, activation, w=1):
        if activation == 'ReLu':
            return w * self.derv_ReLu(x)
        else:
            return w * self.derv_sigmoid(x)

    def dadw(self, x, activation, firstweight=0):
        if firstweight == 1:
            if activation == 'ReLu':
                return self.ReLu(x)
            else:
                return self.sigmoid(x)
        return x

    def feedforward(self, inputs, activations, layers):
        y_pred = []
        affines = []
        val = np.array(inputs)
        for j in range(layers):
            activated_op = []
            n_output = np.dot(val, self.weights[j].T) + self.bias[j]
            if activations[j] == "ReLu":
              for i in n_output:
                affines.append(i)
                activated_op.append(self.ReLu(i))
            else:
              for i in n_output:
                affines.append(i)
                activated_op.append(self.sigmoid(i))
            val = activated_op
        y_pred.append(val)
        return y_pred, affines

    def backpropagation(self, y_pred, y, neurons, affines, layers):
        backprop_layers = neurons[::-1]
        i = 0
        for j in range(backprop_layers[i] * backprop_layers[i + 1]):
            dldy = (y_pred - y)/y_pred(1 - y_pred)
            dyda = self.derv_sigmoid(affines[j+1][j])
            for j in range(i):

            dadw = self.derv_ReLu(affines[i])
            i += 1
            if i > len(backprop_layers) - 1:
                break;











