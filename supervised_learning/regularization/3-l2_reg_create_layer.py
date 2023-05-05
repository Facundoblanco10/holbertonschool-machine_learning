#!/usr/bin/env python3
"""Creates a tensorflow layer that includes L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    @prev is a tensor containing the output of the previous layer
    @n is the number of nodes the new layer should contain
    @activation is the activation function that should be used on the layer
    @lambtha is the L2 regularization parameter
    Returns: the output of the new layer
    """
    # Define the regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
        
    # Create the layer with L2 regularization
    layer = tf.layers.dense(inputs=prev, units=n, activation=activation, kernel_regularizer=regularizer)

    return layer
