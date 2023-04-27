#!/usr/bin/env python3
"""
Creates the training operation for a neural network in tensorflow
using the gradient descent with momentum optimization algorithm
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    @loss is the loss of the network
    @alpha is the learning rate
    @beta1 is the momentum weight
    Returns: the momentum optimization operation
    """
    # Create a MomentumOptimizer object with the
    # specified learning rate and momentum weight
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha,
                                           momentum=beta1)

    # Create the training operation by calling the minimize
    # method on the optimizer object with the loss as an argument
    train_op = optimizer.minimize(loss)

    return train_op
