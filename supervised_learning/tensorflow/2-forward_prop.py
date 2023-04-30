#!/usr/bin/env python3
"""Forward Propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for a neural network.
    Arguments:
    x -- placeholder for the input data
    layer_sizes -- list containing the number of
    nodes in each layer of the network
    activations -- list containing the activation
    functions for each layer of the network
    Returns:
    tensor representing the output of the neural network
    """
    for i in range(len(layer_sizes)):
        n = layer_sizes[i]
        activation = activations[i]
        layer = create_layer(x, n, activation)
        x = layer
    return layer
