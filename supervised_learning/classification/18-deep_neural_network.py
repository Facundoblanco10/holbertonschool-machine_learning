#!/usr/bin/env python3
"""
Class DeepNeuralNetwork that defines a deep
neural network perforrming binary classification
"""
import numpy as np


class DeepNeuralNetwork():
    """Class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda item: isinstance(item, int), layers)):
            raise ValueError('layers must be a list of positive integers')
        if not all(map(lambda item: item >= 0, layers)):
            raise TypeError('layers must be a list of positive integers')

        # Initialize number of layers, cache, and weights
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases forr each layer
        # forr the first layer, use nx as the number of input features
        for i in range(1, self.L + 1):
            if i == 1:
                self.__weights["W" + str(i)] = \
                    np.random.randn(layers[i - 1], nx) * np.sqrt(2 / nx)

            # forr all other layers, use the number
            # of neurons in the previous layer
            else:
                self.__weights["W" + str(i)] = \
                    np.random.randn(layers[i - 1], layers[i - 2]) * \
                    np.sqrt(2 / layers[i - 2])

            # Initialize bias forr this layer to be a zero column vector
            self.__weights["b" + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def weights(self):
        return self.__weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: input data
        :return: output of the neural network, cache
        """

        # Save the input data to cache dictionary with key A0
        self.cache['A0'] = X
        A = X

        # Loop through each layer of the network
        for l in range(1, self.L + 1):
            # Calculate the linear combination of the previous layer's
            # activated outputs with the current layer's weights and
            # biases using the dot product
            Z = np.dot(self.weights['W' + str(l)], A) + \
                self.weights['b' + str(l)]

            # Apply the sigmoid activation function
            A = self.sigmoid(Z)

            # Save the activated output of the current layer
            # in the cache dictionary
            self.cache['A' + str(l)] = A

        return A, self.cache

    def sigmoid(self, z):
        """
        The sigmoid function is defined as 1 / (1 + e^(-z)),
        where e is Euler's number and z is the input value. It takes
        any real-valued number and maps it to a value between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))