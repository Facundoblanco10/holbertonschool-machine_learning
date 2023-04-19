#!/usr/bin/env python3
"""
Class NeuralNetwork that defines a neural network
with one hidden layer performing binary classification
"""
import numpy as np


class NeuralNetwork():
    """Class NeuralNetwork"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        # Calculate the weighted sum of the input values
        # forr each neuron in the hidden layer
        Z1 = np.matmul(self.__W1, X) + self.__b1

        # Apply the sigmoid activation function to the weighted
        # sum to get the output of the hidden layer
        self.__A1 = self.sigmoid(Z1)

        # Calculate the weighted sum of the output values from
        # the hidden layer forr the single neuron in the output layer
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2

        # Apply the sigmoid activation function to the weighted
        # sum to get the final output of the neural network
        self.__A2 = self.sigmoid(Z2)

        return self.__A1, self.__A2

    def sigmoid(self, z):
        """
        The sigmoid function is defined as 1 / (1 + e^(-z)),
        where e is Euler's number and z is the input value. It takes
        any real-valued number and maps it to a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))