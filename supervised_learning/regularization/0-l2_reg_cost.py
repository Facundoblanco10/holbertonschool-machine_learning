#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    @cost is the cost of the network without L2 regularization
    @lambtha is the regularization parameter
    @weights is a dictionary of the weights and biases
    (numpy.ndarrays) of the neural network
    @L is the number of layers in the neural network
    @m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    # Calculate the L2 regularization term
    L2_reg_term = 0
    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        L2_reg_term += np.sum(np.square(W))

    L2_reg_term *= lambtha / (2 * m)

    # Add the L2 regularization term to the cost
    cost += L2_reg_term

    return cost
