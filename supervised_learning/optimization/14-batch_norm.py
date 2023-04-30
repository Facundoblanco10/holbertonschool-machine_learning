#!/usr/bin/env python3
"""
Creates a batch normalization layer
for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    @prev is the activated output of the previous layer
    @n is the number of nodes in the layer to be created
    @activation is the activation function that
    should be used on the output of the layer
    Returns: a tensor of the activated output for the layer
    """
    # Initialize the base layer with 'Dense' function from tensorflow
    # The 'prev' input is passed through a dense layer with n nodes
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=k_init,
                            use_bias=False)(prev)

    # Initialize trainable parameters 'gamma' and 'beta' as vectors of 1
    # and 0 respectively, with shape (1, n)
    gamma = tf.Variable(initial_value=tf.ones(shape=(1, n)), name="gamma")
    beta = tf.Variable(initial_value=tf.zeros(shape=(1, n)), name="beta")

    # Calculate the batch mean and variance of the previous layer
    # using the tf.nn.moments() function
    mean, variance = tf.nn.moments(layer, axes=[0])

    # Create a batch normalization layer using the
    # tf.nn.batch_normalization() function
    # with gamma, beta, mean, variance and epsilon=1e-8 as arguments
    # Apply the activation function to
    # the output of the batch normalization layer
    # The final result is a tensor of the activated output for the layer
    layer_norm = tf.nn.batch_normalization(layer,
                                           mean,
                                           variance,
                                           beta,
                                           gamma,
                                           1e-8)
    output = activation(layer_norm)
    return output
