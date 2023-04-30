#!/usr/bin/env python3
"""Adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds arrays"""
    if len(arr1) != len(arr2):
        return None
    result = list(map(lambda x, y: x + y, arr1, arr2))

    return result
