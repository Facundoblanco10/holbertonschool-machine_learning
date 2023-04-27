#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """
    @data is the list of data to calculate the moving average of
    @beta is the weight used for the moving average
    Returns: a list containing the moving averages of data
    """
    # Initialize variables
    moving_avg = []
    v = 0.0  # exponentially weighted average of previous values

    # correction factor for bias due to initial v value
    bias_correction = 1.0

    # Calculate the exponentially weighted moving average
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        moving_avg.append(v / (bias_correction - beta**(i + 1)))

    return moving_avg
