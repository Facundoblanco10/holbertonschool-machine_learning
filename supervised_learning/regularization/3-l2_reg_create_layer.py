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
    # Initlializes the init variable with the variance scaling initializer
    # from TensorFlow's contrib module
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Initizalizes the regularizer variable
    # with the L2 regularization function.
    # The lambtha parameter specifies the strength of the regularization
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)

    # Creates a dense layer with n units, using the specified activation
    # function and input from the previous layer prev. The layer's
    # weights are initialized using the init variable and regularized
    # using the regularizer variable
    layer = tf.layers.dense(units=n,
                            activation=activation,
                            inputs=prev,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)

    return layer
