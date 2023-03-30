#!/usr/bin/env python3
"""Element-wise addition, subtraction,"""
""" division and multiplication"""


def np_elementwise(mat1, mat2):
    """Elementwise function"""
    add = mat1 + mat2

    sub = mat1 - mat2

    mul = mat1 * mat2

    div = mat1 / mat2

    return (add, sub, mul, div)