#!/usr/bin/env python3
"""Matrix Transposition"""


def matrix_transpose(matrix):
    """Transpose a matrix"""
    transpose = []
    for idx, i in enumerate(matrix):
        trix = []
        for j in matrix:
            if idx < len(j):
                trix.append(j[idx])
        if len(trix):
            transpose.append(trix)
    return (transpose)
