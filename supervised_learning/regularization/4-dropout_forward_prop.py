#!/usr/bin/env python3
"""Conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    @X is a numpy.ndarray of shape (nx, m) containing the
    input data for the network
        nx is the number of input features
        m is the number of data points
    @weights is a dictionary of the weights and biases of the neural network
    @L the number of layers in the network
    @keep_prob is the probability that a node will be kept
    Returns: a dictionary containing the outputs of each layer and the
    dropout mask used on each layer (see example for format)
    """
