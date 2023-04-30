#!/usr/bin/env python3
"""Concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices"""
    result = []
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        for row1, row2 in zip(mat1, mat2):
            row_res = row1 + row2
            result.append(row_res)
        return result
    else:
        return None
