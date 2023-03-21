#!usr/bin/env python3
"""Perform a matrix multiplication"""


def mat_mul(mat1, mat2):
    """Perform a 2D matrix multiplication"""
    m1, n1 = len(mat1), len(mat1[0])
    m2, n2 = len(mat2), len(mat2[0])

    if n1 != m2:
        return None

    result = [[0 for j in range(n2)] for i in range(m1)]

    for i in range(m1):
        for j in range(n2):
            for k in range(n1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
