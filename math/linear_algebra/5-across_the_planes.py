#!/usr/bin/env python3
"""Adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds arrays"""
    if (len(mat1) != len(mat2) or
            len(mat1[0]) != len(mat2[0]) or
            len(mat1) != len(mat2)):
        return None
    result = []
    for row1, row2 in zip(mat1, mat2):
        row_result = list(map(lambda x, y: x + y, row1, row2))
        result.append(row_result)

    return result
