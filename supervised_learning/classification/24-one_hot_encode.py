#!/usr/bin/env python3
"""Converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """
        Converts a numeric label vector into a one-hot matrix
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0 or \
            not isinstance(classes, int) or classes <= np.amax(Y):
        return None

    #  initializes a numpy array one_hot with classes rows
    # and Y.shape[0] columns with all elements set to 0.
    one_hot = np.zeros((classes, Y.shape[0]))

    # sets the value at index [Y[i], i] of one_hot to 1 for all i in Y.
    one_hot[Y, np.arange(Y.shape[0])] = 1

    return one_hot
