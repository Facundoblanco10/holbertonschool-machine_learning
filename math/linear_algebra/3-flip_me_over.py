#!/usr/bin/env python3
"""Matrix Transposition"""


def matrix_transpose(matrix):
    """Transpose a matrix"""
    rows = len(matrix)
    cols = len(matrix[0])

    transposed_matrix = [[0 for j in range(rows)] for i in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix
