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
        self.W = np.random.normal(size=(nx,))
        self.b = 0
        self.A = 0
