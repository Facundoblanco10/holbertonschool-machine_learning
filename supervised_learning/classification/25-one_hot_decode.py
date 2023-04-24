#!/usr/bin/env python3
"""Converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """
        Converts a one-hot matrix into a vector of labels
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    m = one_hot.shape[1]

    # get the index of the maximum value along the axis 0 of the one_hot
    # array, which returns an array of indices. It then reshapes this
    # array to be a one-dimensional array of size m using the reshape function
    return np.argmax(one_hot, axis=0).reshape((m,))
