#!/usr/bin/env python3
"""Calculate Sigma i goes from"""
"""1 to n and we sum i^2"""


def summation_i_squared(n):
    """sum i squared"""
    if type(n) != int or n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1)/6)
