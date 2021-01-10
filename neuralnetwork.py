import numpy as np
from sklearn import datasets


np.random.seed(0)

#Data to feed the Neural Network
data = datasets.make_circles(n_samples=100, shuffle=True, noise=0.10, random_state=None, factor=0.25) #Generate annulus data

x1 = data[0][:,0] #Features
x2 = data[0][:,1]

inputs = [] #Zip the inputs together i.e [x1, x2]
for x1, x2 in zip(x1, x2):
  inputs.append([x1, x2])

y = data[1] #Ground truth

class NeuralNetwork(object):

    def __init__(self, x, y): #Constructor to initialise the neural network

        #Initialisation
        self.layers = 4
        self.neurons = [2, 2, 2, 1]
        self.activations = ['ReLu', 'ReLu', 'sigmoid']
        self.x = x
        self.y = y
        self.weights = []
        self.weight_list = []
        self.bias = []
        #Initialisation

        #Initialise the weights and biases
        for i in range(len(self.neurons) - 1):
            self.weights.append(np.random.normal(0, np.sqrt(2/self.neurons[i]), (self.neurons[i+1], self.neurons[i]))) #He initialisation
            self.bias.append(np.random.normal(self.neurons[i+1], 1)) #Bias for the layers

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


    def feedforward(self, inputs):
        y_pred = []
        affines = []
        val = np.array(inputs)
        for j in range(self.layers - 1):
            activated_op = []
            n_output = np.dot(val, self.weights[j].T) + self.bias[j]
            if self.activations[j] == "ReLu":
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

    def backprop(self, inputs, y_pred, affines, weight_list):
        for items in self.weights:
            for item in items:
                for weight in item:
                    self.weight_list.append(weight)
        affine_vals = affines[::-1]
        weight_list = weight_list[::-1]
        dldy_val = self.dldy(y_pred, y[0])
        dyda_val = self.derv_sigmoid(affine_vals[0])
        dadw10 = dldy_val * dyda_val * self.ReLu(affine_vals[1])
        dadw9 = dldy_val * dyda_val * self.ReLu(affine_vals[2])
        dadw8 = dldy_val * dyda_val * self.dada(affine_vals[1], self.activations[1], w=weight_list[0]) * self.ReLu(affine_vals[3])
        dadw7 = dldy_val * dyda_val * self.dada(affine_vals[1], self.activations[1], w=weight_list[0]) * self.ReLu(affine_vals[4])
        dadw6 = dldy_val * dyda_val * self.dada(affine_vals[2], self.activations[1], w=weight_list[1]) * self.ReLu(affine_vals[3])
        dadw5 = dldy_val * dyda_val * self.dada(affine_vals[2], self.activations[1], w=weight_list[1]) * self.ReLu(affine_vals[4])
        dadw4 = dldy_val * dyda_val * ((self.dada(affine_vals[2], self.activations[2], w=weight_list[1]) * weight_list[4]) + (
                    self.dada(affine_vals[1], self.activations[2], w=weight_list[0]) * weight_list[2])) * self.ReLu(affine_vals[3]) * \
                inputs[1]
        dadw3 = dldy_val * dyda_val * ((self.dada(affine_vals[2], self.activations[2], w=weight_list[1]) * weight_list[4]) + (
                    self.dada(affine_vals[1], self.activations[2], w=weight_list[0]) * weight_list[2])) * self.ReLu(affine_vals[3]) * \
                inputs[0]
        dadw2 = dldy_val * dyda_val * ((self.dada(affine_vals[2], self.activations[2], w=weight_list[1]) * weight_list[5]) + (
                    self.dada(affine_vals[1], self.activation[2], w=weight_list[0]) * weight_list[3])) * self.ReLu(affine_vals[4]) * \
                inputs[1]
        dadw1 = dldy_val * dyda_val * ((self.dada(affine_vals[2], self.activations[2], w=weight_list[1]) * weight_list[5]) + (
                    self.dada(affine_vals[1], self.activation[2], w=weight_list[0]) * weight_list[3])) * self.ReLu(affine_vals[4]) * \
                inputs[0]

        return [dadw1, dadw2, dadw3, dadw4, dadw5, dadw6, dadw7, dadw8, dadw9, dadw10]

    def gradient_descent(self, gradients, learning_rate):
        counter = 0
        for i in range(len(self.neurons) - 1):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = self.weights[i][j][k] - learning_rate * gradients[counter]
                    counter += 1

    def train(self, x,y,batch_size=10, epochs=10, lr=0.01):
        for e in range(epochs):
            i = 0
            while i<len(y):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                i = i+batch_size

                output, affines = self.feedforward(x_batch)
                gradients = self.backprop(y_batch, output_from_layers, intermediate_inputs)

                self.weights = [w+lr*dweight for w,dweight in zip(self.weights, dw)]
                self.biases = [w+lr*dbias for w,dbias in zip(self.biases, db)]

                print("loss = {}".format(np.linalg.norm(intermediate_inputs[-1]-y_batch)))

















