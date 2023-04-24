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

    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1

    return one_hot
