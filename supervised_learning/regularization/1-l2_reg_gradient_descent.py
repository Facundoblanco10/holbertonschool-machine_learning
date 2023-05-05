#!/usr/bin/env python3
"""
Updates the weights and biases of a neuralnetwork
using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    @Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points
    @weights is a dictionary of the weights and biases of the neural network
    @cache is a dictionary of the outputs of each layer of the neural network
    @alpha is the learning rate
    @lambtha is the L2 regularization parameter
    @L is the number of layers of the network
    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    """
    m = Y.shape[1]

    # Backward pass
    dZ = cache['A' + str(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        # Calculates the inverse of the number of data points m
        m_inv = 1 / m

        # Computes the gradient of the weight matrix W for the
        # current layer using backpropagation with L2 regularization.
        dW = m_inv * np.dot(dZ, A_prev.T) + (lambtha / m) * W

        # Computes the gradient of the bias vector b
        # for the current layer using backpropagation.
        db = m_inv * np.sum(dZ, axis=1, keepdims=True)

        # Computes the gradient of the activations A
        # for the previous layer using backpropagation
        dZ = np.dot(W.T, dZ) * (1 - np.square(A_prev))

        # Update weights and biases
        W -= alpha * dW
        b -= alpha * db
