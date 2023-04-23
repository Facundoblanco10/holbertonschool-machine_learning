#!/usr/bin/env python3
"""
Class DeepNeuralNetwork that defines a deep
neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork():
    """Class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda item: isinstance(item, int) and
                       item > 0, layers)):
            raise ValueError('layers must be a list of positive integers')

        # Initialize number of layers, cache, and weights
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialize weights and biases for each layer
        for i in range(self.L):
            # For the first layer, use nx as the number of input features
            if i == 0:
                self.weights["W" + str([i + 1])] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)

            # For all other layers, use the number
            # of neurons in the previous layer
            else:
                self.weights["W" + str([i + 1])] = \
                    np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Initialize bias for this layer to be a zero column vector
            self.weights["b" + str([i + 1])] = np.zeros((layers[i], 1))
