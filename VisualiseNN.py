#Include necessary packages

#Tool imports
import numpy as np
import pandas as pd
import networkx as nx

#Dataset imports
from sklearn.model_selection import train_test_split
from sklearn import datasets

#Keras imports
import tensorflow as tf
tf.random.set_seed(59)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

#Graphic imports
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#Generate annulus data using scikit's datasets.make_circle
data = datasets.make_circles(n_samples=100, shuffle=True, noise=0.10, random_state=None, factor=0.25)

#Seperate into features and targets

X = data[0] #features
y = data[1] #targets

#Split into training and test data:
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


activations = [] #Store outputs of each layer after every fifth epoch
weights = [] #Store weights of each layer after every fifth epoch]


#Class for call backs
class model_monitor(Callback):

    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.N == 0:
            for layer in self.model.layers:
                weight = layer.get_weights()
                outputs = layer.output # all layer outputs
                weights.append(weight)
                print(outputs)
        self.epoch += 1



#Build the model
model = Sequential()


model.add(Dense(5, input_shape = (67, 2), activation = None, name = 'input'))
model.add(Dense(3, activation = 'relu'))
# Add the output layer with one neuron and linear activation
model.add(Dense(1, activation = 'sigmoid', name = 'output'))

# View the model summary
model.summary()

#Compile the model
model.compile(optimizers.SGD(0.1), loss = 'binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

#Fit the data
history = model.fit(x_train, y_train, epochs = 100, validation_data=(x_test, y_test), callbacks=[model_monitor(model, 5)],  batch_size = 1, verbose = 1)

#Inference
accuracy = model.evaluate(x_test, y_test)
print('****************weights*****************')
print(weights[0])


