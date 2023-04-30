#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if type(poly) != list or type(C) != int:
        return None
    if len(poly) == 0:
        return None
    result = [C]
    if poly == [0]:
        return result
    idx = 1
    for i in poly:
        result.append(i / idx)
        if result[idx] % 1 == 0:
            result[idx] = int(result[idx])
        idx += 1

    return result
