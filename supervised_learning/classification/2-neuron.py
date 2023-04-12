#!/usr/bin/env python3
"""Class Neuron that defines a single neuron"""
import numpy as np


class Neuron():
    """Neuron class that performs binary classification"""

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.__A

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
