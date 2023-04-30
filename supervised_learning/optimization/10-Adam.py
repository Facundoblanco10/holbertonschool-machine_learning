#!/usr/bin/env python3
"""
creates the training operation for a neural network
in tensorflow using the Adam optimization algorithm
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    @loss is the loss of the network
    @alpha is the learning rate
    @beta1 is the weight used for the first moment
    @beta2 is the weight used for the second moment
    @epsilon is a small number to avoid division by zero
    Returns: the Adam optimization operation
    """
    # Create global step variable
    global_step = tf.Variable(0, trainable=False)

    # Define Adam optimizer with given hyperparameters
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)

    # Compute gradients and apply them using the Adam optimizer
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars,
                                         global_step=global_step)

    return train_op
