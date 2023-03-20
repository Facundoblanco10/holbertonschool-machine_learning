#!/usr/bin/env python3
"""Shape a matrix"""


def matrix_shape(matrix):
    shape = []
    shape.append(len(matrix))
    while matrix[0] and type(matrix[0]) != int:
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return (shape)
