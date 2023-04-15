#!/usr/bin/env python3
"""
Class DeepNeuralNetwork that defines a deep
neural network performing binary classification
"""


class DeepNeuralNetwork():
    """Class DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        