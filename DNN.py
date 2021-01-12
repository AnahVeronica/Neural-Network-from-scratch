import numpy as np
from sklearn import datasets
np.random.seed(0)

class NeuralNetwork():

    def __init__(self):
        self.inputs = 2                #No of inputs
        self.hidden_layers = [2]    #No of hidden layers
        self.output_node = 1      #No of output nodes
        self.activator = ['ReLu', 'Sigmoid']
        self.weights = []         #No of weights
        self.weight_list = []     #query the weights
        self.bias = []            #No of bias
        self.layers = [self.inputs] + self.hidden_layers + [self.output_node] #No. of hidden layers
        # Create zero array to store activations
        self.activation = [np.zeros(self.layers[i]) for i in range(len(self.layers))]
        #Create zero array to store derivatives
        self.derivatives = [np.zeros((self.layers[i], self.layers[i + 1])) for i in range(len(self.layers) - 1)]


        #Initialise the weights
        for i in range(len(self.layers) - 1):
            weight = np.random.rand(self.layers[i], self.layers[i + 1])
            self.weights.append(weight)
        self.bias = np.random.rand(self.layers[i + 1], 1)

        #Fill in weight_list:
        for items in self.weights:
            for item in items:
                for weight in item:
                    self.weight_list.append(weight)

    #Activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def ReLu(self, x):
        for i in range(len(x)):
            x[i] = max(0, x[i])
        return x

    #Derivative of the activation functions
    def derv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def dldy(self, y_pred, y):
        return (y_pred - y)/y_pred*(1 - y_pred)

    #Feedforward mechanism

    def feedforward(self, inputs):
        activated_ip = inputs

        self.activation[0] = activated_ip

        for i, w in enumerate(self.weights):
            affine = np.dot(activated_ip, w)

            activated_ip = self.sigmoid(affine) if self.activator[i] == 'Sigmoid' else self.ReLu(affine)

            self.activation[i + 1] = activated_ip
        return activated_ip[0] #returns the output y for the given weights.

    def cross_entropy_loss(self, y_pred, y):
        return -(y * np.log(y_pred) + ((1 - y) * np.log(1 - y_pred)))

    def back_propagate(self, y_pred, y):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activation[i+1]

            # apply sigmoid derivative function
            delta = self.dldy(y_pred, y) * self.derv_sigmoid(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activation[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = np.array(current_activations).reshape(np.array(current_activations).shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights -= derivatives * learningRate

    def train(self, inputs, target, epochs, learning_rate):
        for i in range(epochs):
            error_avg = 0
            for j, input in enumerate(inputs):
                ground_truth = target[j]
                prediction = self.feedforward(input)
                self.back_propagate(prediction, ground_truth)
                self.gradient_descent(learning_rate)
                error_avg += self.cross_entropy_loss(prediction, ground_truth)
            print("Error: {} at epoch {}".format(error_avg, i + 1))
        print("Training complete!")
        print("=====")







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
    nn = NeuralNetwork()
    nn.train(inputs, y, 100, 0.01)

