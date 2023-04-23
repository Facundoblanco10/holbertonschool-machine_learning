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
